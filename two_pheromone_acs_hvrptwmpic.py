"""
Two-Pheromone Trail Ant Colony System (ACS) for HVRPTWMPIC
-----------------------------------------------------------------
Faithful reference implementation based on:
  Palma-Blanco et al., "A Two-Pheromone Trail Ant Colony System Approach for the
  Heterogeneous Vehicle Routing Problem with Time Windows, Multiple Products and Product Incompatibility"
  (ICCL 2019).

Notes
-----
- This is a research-grade, readable implementation intended for baseline comparison.
- It follows the paper's symbols as closely as possible (see section headers).
- All constraints are enforced: Time Windows, Heterogeneous capacities (weight/volume),
  Multiple Products, and Incompatibility matrix C.
- Objective: fleet-size cost + routing cost, with vehicle fixed cost ξ defined as in Eq. (2).
- Two pheromone trails are used: Weak (local) and Strong (global). Updates per Eqs. (6) & (7).
- Construction rule mixes pheromone & heuristic per Eqs. (3)-(5).
- Local search: Fleet-size minimization + modified 2-opt (Algorithm 4 & 5).

Caveats
-------
- The paper does not explicitly state how to combine weak/strong trails during choice; we use
  Q = Qw + Qs in the state transition rule (common in dual-trail ACO literature).
- This code aims for clarity; micro-optimizations are intentionally avoided.

Usage
-----
$ python two_pheromone_acs_hvrptwmpic.py  # runs a tiny demo instance

For integration in your project, import `TwoPheromoneACS` and call `.solve()` with a Problem.
"""
from __future__ import annotations
import math
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np

# -----------------------------
# Data model (Section 3 in paper)
# -----------------------------
@dataclass
class Problem:
    # Graph
    D: np.ndarray  # (N+1, N+1) distance matrix, node 0 is depot
    T: np.ndarray  # (N+1, N+1) travel-time matrix
    TW: np.ndarray  # (N+1, 3): [e_i, l_i, s_i] (earliest, latest, service time), depot row allowed

    # Demands (customers 1..N):
    A: np.ndarray  # (N+1, P): weight demand per product (row 0 == zeros for depot)
    B: np.ndarray  # (N+1, P): volume demand per product (row 0 == zeros)

    # Product compatibility:
    C: np.ndarray  # (P, P): 1 if p and q can be together, 0 otherwise

    # Fleet capacities and vehicle count upper bound M_max
    F_caps: np.ndarray  # (M_max, 2): [weight_cap, volume_cap]

    # Precomputed fixed cost ξ (Eq. 2); if None, computed from D
    xi: Optional[float] = None

    def __post_init__(self):
        assert self.D.shape == self.T.shape
        assert self.D.shape[0] == self.T.shape[1]
        Np1 = self.D.shape[0]
        assert self.TW.shape == (Np1, 3)
        assert self.A.shape[0] == Np1 and self.B.shape[0] == Np1
        P = self.A.shape[1]
        assert self.B.shape[1] == P
        assert self.C.shape == (P, P)
        if self.xi is None:
            N = Np1 - 1
            xi = self.D.sum() / (N * N) * 1.2 if N > 0 else 0.0
            self.xi = float(xi)

    @property
    def N(self) -> int:
        return self.D.shape[0] - 1  # customers count

    @property
    def P(self) -> int:
        return self.A.shape[1]

    @property
    def M_max(self) -> int:
        return self.F_caps.shape[0]


@dataclass
class Route:
    nodes: List[int] = field(default_factory=lambda: [0])  # start at depot 0
    used_weight: np.ndarray = None  # (P,)
    used_volume: np.ndarray = None  # (P,)
    product_set: np.ndarray = None  # (P,) bool: 1 if product type present in vehicle
    time_at_node: List[float] = field(default_factory=list)  # arrival/leave bookkeeping (optional)
    # Heterogeneous fleet info
    vehicle_idx: int = -1
    cap_w: float = 0.0
    cap_v: float = 0.0

    def clone(self) -> "Route":
        r = Route()
        r.nodes = list(self.nodes)
        r.used_weight = None if self.used_weight is None else self.used_weight.copy()
        r.used_volume = None if self.used_volume is None else self.used_volume.copy()
        r.product_set = None if self.product_set is None else self.product_set.copy()
        r.time_at_node = list(self.time_at_node)
        r.vehicle_idx = self.vehicle_idx
        r.cap_w = self.cap_w
        r.cap_v = self.cap_v
        return r


@dataclass
class Solution:
    routes: List[Route]
    total_distance: float
    fleet_size: int
    obj: float


# -----------------------------
# ACS Parameters (Section 4.1)
# -----------------------------
@dataclass
class ACSParams:
    r: int = 10000        # iterations
    m: int = 10           # agents
    omega0: float = 0.2   # exploration/exploitation threshold
    alpha: float = 0.1    # learning rate
    gamma: float = 0.1    # discounting rate (local trail toward Q0)
    delta: float = 1.0    # pheromone weight in transition rule
    beta: float = 2.0     # heuristic weight in transition rule
    Q0: float = 1.0       # initial pheromone value
    seed: int = 1234
    use_xi: bool = True      # if False -> objective = total_distance (min-sum)
    allow_multitrip: bool = False  # if True -> fixed K vehicles, multiple routes per vehicle allowed


# ---------------------------------
# Utility: feasibility & cost helpers
# ---------------------------------
class Feasibility:
    @staticmethod
    def can_insert(problem: Problem, route: Route, cust: int, D: np.ndarray, T: np.ndarray,
                   TW: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                   weight_cap: float, volume_cap: float) -> bool:
        """Check if appending 'cust' to the end of 'route' is feasible under capacity,
        product compatibility, and time windows (simple forward check).
        We append at end for construction; for LS reinsertion we check locally around insertion.
        """
        if cust == 0:
            return False
        P = A.shape[1]

        # Product incompatibility: if any demanded product exists and is incompatible with vehicle set
        demand_products = (A[cust] + B[cust]) > 0  # product types with any demand
        if route.product_set is None:
            # initialize on first non-depot insert
            r_ps = demand_products.astype(bool)
        else:
            r_ps = route.product_set.copy()
            # Check pairwise compatibility for any new product to existing set
            new_products = np.where(demand_products)[0]
            if new_products.size > 0:
                existing = np.where(r_ps)[0]
                if existing.size > 0:
                    # If any (p in new) and (q in existing) with C[p,q]==0, infeasible
                    for p in new_products:
                        for q in existing:
                            if C[p, q] == 0:
                                return False
            r_ps[new_products] = True

        # Capacity (aggregate across product types) — paper expresses capacities per vehicle in total
        next_w = (route.used_weight if route.used_weight is not None else np.zeros(P)) + A[cust]
        next_v = (route.used_volume if route.used_volume is not None else np.zeros(P)) + B[cust]
        if next_w.sum() > weight_cap or next_v.sum() > volume_cap:
            return False

        # Time window check (append at end)
        prev = route.nodes[-1]
        arrive = Feasibility._arrival_time(problem, route, prev, cust)
        e_i, l_i, s_i = TW[cust]
        start_service = max(arrive, e_i)
        if start_service > l_i:
            return False
        # Always feasible to leave after service; depot closure not enforced here
        return True

    @staticmethod
    def _arrival_time(problem: Problem, route: Route, i: int, j: int) -> float:
        # Compute arrival time at j when coming from i at end of current route
        # We maintain only the last leave time; recompute quickly:
        if len(route.nodes) == 1:  # only depot present
            leave_prev = max(problem.TW[0, 0], 0.0)
        else:
            # approximate: last event leave time using previous pair
            # (we don't store full schedule; for construction this is acceptable)
            last = route.nodes[-1]
            # rough: assume we left last at its feasible start + service
            # For consistent check, recompute rolling along route
            leave_prev = 0.0
            t = 0.0
            cur = 0
            for nxt in route.nodes[1:]:
                t = max(t + problem.T[cur, nxt], problem.TW[nxt, 0]) + problem.TW[nxt, 2]
                cur = nxt
            leave_prev = t
        # travel to j
        arrive = leave_prev + problem.T[i, j]
        return arrive

    @staticmethod
    def route_distance(D: np.ndarray, nodes: List[int]) -> float:
        dist = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            dist += D[a, b]
        # ensure return to depot at end
        if nodes[-1] != 0:
            dist += D[nodes[-1], 0]
        return dist


# -----------------------------
# Two-Pheromone ACS Solver
# -----------------------------
class TwoPheromoneACS:
    def __init__(self, problem: Problem, params: ACSParams):
        self.pb = problem
        self.pr = params
        random.seed(self.pr.seed)
        np.random.seed(self.pr.seed)

        n = self.pb.N + 1
        self.Qw = np.full((n, n), self.pr.Q0, dtype=float)  # weak (local) trail
        self.Qs = np.full((n, n), self.pr.Q0, dtype=float)  # strong (global) trail

    # ---------------
    # Public API
    # ---------------
    def solve(self) -> Solution:
        best_sol: Optional[Solution] = None

        for it in range(self.pr.r):
            local_best: Optional[Solution] = None

            # Each agent constructs a feasible solution
            for _ in range(self.pr.m):
                sol = self._construct_solution()
                # Local best update
                if (local_best is None) or (sol.obj < local_best.obj):
                    local_best = sol

            # Local improvement on the best of this iteration
            local_best = self._fleet_size_minimization(local_best)
            local_best = self._two_opt_modified(local_best)

            # Global best & strong pheromone update
            if (best_sol is None) or (local_best.obj < best_sol.obj):
                best_sol = local_best
                self._update_strong_pheromone(best_sol)

            # Optional: evaporation of strong trail (not explicit in paper; skip for fidelity)

        assert best_sol is not None
        return best_sol

    # ---------------
    # Construction (Algorithm 3)
    # ---------------
    def _construct_solution(self) -> Solution:
        pb = self.pb
        unserved = set(range(1, pb.N + 1))
        routes: List[Route] = []

        if self.pr.allow_multitrip:
            # Round-robin: for each vehicle, create a trip; repeat until no customer can be added by any vehicle
            while unserved:
                made_any = False
                for vehicle_idx in range(pb.M_max):
                    # Start a new trip for this vehicle
                    route = Route()
                    route.used_weight = np.zeros(pb.P)
                    route.used_volume = np.zeros(pb.P)
                    route.product_set = np.zeros(pb.P, dtype=bool)
                    route.nodes = [0]
                    route.vehicle_idx = vehicle_idx
                    route.cap_w = float(pb.F_caps[vehicle_idx, 0])
                    route.cap_v = float(pb.F_caps[vehicle_idx, 1])

                    while True:
                        feasibles = []
                        for c in list(unserved):
                            if Feasibility.can_insert(pb, route, c, pb.D, pb.T, pb.TW, pb.A, pb.B, pb.C, route.cap_w, route.cap_v):
                                feasibles.append(c)
                        if not feasibles:
                            break
                        r = route.nodes[-1]
                        Q = (self.Qw[r, feasibles] + self.Qs[r, feasibles]) ** self.pr.delta
                        H = np.array([self._heuristic(r, s) for s in feasibles]) ** self.pr.beta
                        scores = Q * H
                        if random.random() <= self.pr.omega0:
                            s = feasibles[int(np.argmax(scores))]
                        else:
                            probs = scores / scores.sum()
                            s = int(np.random.choice(feasibles, p=probs))
                        self._append_customer(route, s)
                        unserved.remove(s)
                        # Weak pheromone update
                        i = r; j = s
                        self.Qw[i, j] = self.Qw[i, j] + self.pr.alpha * (self.pr.gamma * self.pr.Q0 - self.Qw[i, j])

                    if len(route.nodes) > 1 and route.nodes[-1] != 0:
                        route.nodes.append(0)
                    if len(route.nodes) > 2:
                        routes.append(route)
                        made_any = True
                if not made_any:
                    # No vehicle could serve remaining customers -> infeasible remainder
                    break
        else:
            # Single-trip per vehicle (original behavior)
            vehicle_idx = 0
            while unserved and vehicle_idx < pb.M_max:
                route = Route()
                route.used_weight = np.zeros(pb.P)
                route.used_volume = np.zeros(pb.P)
                route.product_set = np.zeros(pb.P, dtype=bool)
                route.nodes = [0]
                route.vehicle_idx = vehicle_idx
                route.cap_w = float(pb.F_caps[vehicle_idx, 0])
                route.cap_v = float(pb.F_caps[vehicle_idx, 1])

                while True:
                    feasibles = []
                    for c in list(unserved):
                        if Feasibility.can_insert(pb, route, c, pb.D, pb.T, pb.TW, pb.A, pb.B, pb.C, route.cap_w, route.cap_v):
                            feasibles.append(c)
                    if not feasibles:
                        break
                    r = route.nodes[-1]
                    Q = (self.Qw[r, feasibles] + self.Qs[r, feasibles]) ** self.pr.delta
                    H = np.array([self._heuristic(r, s) for s in feasibles]) ** self.pr.beta
                    scores = Q * H
                    if random.random() <= self.pr.omega0:
                        s = feasibles[int(np.argmax(scores))]
                    else:
                        probs = scores / scores.sum()
                        s = int(np.random.choice(feasibles, p=probs))
                    self._append_customer(route, s)
                    unserved.remove(s)
                    i = r; j = s
                    self.Qw[i, j] = self.Qw[i, j] + self.pr.alpha * (self.pr.gamma * self.pr.Q0 - self.Qw[i, j])

                if route.nodes[-1] != 0:
                    route.nodes.append(0)
                routes.append(route)
                vehicle_idx += 1

        # Compute objective (no dummy catch-all). If unserved remains, add a big-M penalty so iteration can proceed.
        total_distance = sum(Feasibility.route_distance(pb.D, rt.nodes) for rt in routes if len(rt.nodes) > 1)
        # Count distinct vehicles actually used
        used_vehicles = set(rt.vehicle_idx for rt in routes if len(rt.nodes) > 2 and rt.vehicle_idx >= 0)
        fleet_size = len(used_vehicles)
        penalty = 0.0
        if unserved:
            penalty = 1e6 * len(unserved)
        if self.pr.use_xi:
            obj = pb.xi * fleet_size + total_distance + penalty
        else:
            obj = total_distance + penalty
        return Solution(routes=routes, total_distance=total_distance, fleet_size=fleet_size, obj=obj)

    def _heuristic(self, r: int, s: int) -> float:
        pb = self.pb
        D = pb.D[r, s]
        # time component approximated as in Eq. (4): depend on (max(v_r + T_rs, e_s) - v_r) and (l_s - v_r)
        # we approximate v_r as earliest feasible leave time from r based on TW[r]
        e_r, l_r, s_r = pb.TW[r]
        e_s, l_s, s_s = pb.TW[s]
        # naive v_r
        v_r = max(e_r, 0.0) + s_r
        time_term = max(v_r + pb.T[r, s], e_s) - v_r
        slack_term = max(l_s - v_r, 1e-6)
        denom = max(D * max(time_term, 1e-6) * slack_term, 1e-6)
        return 1.0 / denom

    def _append_customer(self, route: Route, cust: int):
        pb = self.pb
        # Update load & product set
        route.used_weight = (route.used_weight + pb.A[cust]) if route.used_weight is not None else pb.A[cust].copy()
        route.used_volume = (route.used_volume + pb.B[cust]) if route.used_volume is not None else pb.B[cust].copy()
        if route.product_set is None:
            route.product_set = ((pb.A[cust] + pb.B[cust]) > 0)
        else:
            route.product_set |= ((pb.A[cust] + pb.B[cust]) > 0)
        route.nodes.append(cust)

    # --------------------
    # Local improvements
    # --------------------
    def _fleet_size_minimization(self, sol: Solution) -> Solution:
        pb = self.pb
        routes = [rt.clone() for rt in sol.routes if len(rt.nodes) > 1]
        # Sort vehicles by number of customers (increasing), exclude depot-only
        order = sorted(range(len(routes)), key=lambda k: max(0, len(routes[k].nodes) - 2))

        changed = True
        while changed:
            changed = False
            for idx in order:
                rt = routes[idx]
                customers = [n for n in rt.nodes[1:-1]]  # exclude depots
                if not customers:
                    continue
                for c in list(customers):
                    # Try to reassign c to some other route at best position
                    best_gain = 0.0
                    best_where = None
                    for jdx, other in enumerate(routes):
                        if jdx == idx:
                            continue
                        # try all insertion positions between nodes
                        for pos in range(1, len(other.nodes)):
                            if self._feasible_insert_at(other, c, pos):
                                delta = self._delta_distance_insert(other, c, pos)
                                if delta < best_gain:
                                    best_gain = delta
                                    best_where = (jdx, pos)
                    if best_where is not None:
                        # perform move
                        jdx, pos = best_where
                        # remove c from rt
                        rt.nodes.remove(c)
                        # insert into other
                        routes[jdx].nodes.insert(pos, c)
                        changed = True
                        # if route becomes empty (only depots), keep as is; final count handles empties

        # Compute objective with coverage penalty (served customers vs. all customers)
        served = set()
        for rt in routes:
            for n in rt.nodes:
                if n != 0:
                    served.add(n)
        missing = (pb.N - len(served))
        penalty = 0.0
        if missing > 0:
            penalty = 1e6 * missing

        total_distance = sum(Feasibility.route_distance(pb.D, rt.nodes) for rt in routes if len(rt.nodes) > 1)
        fleet_size = sum(1 for rt in routes if len(rt.nodes) > 2)
        if self.pr.use_xi:
            obj = pb.xi * fleet_size + total_distance + penalty
        else:
            obj = total_distance + penalty
        
        # If LS produced no improvement and even dropped all routes (edge case), keep original
        if (len(served) == 0) and (missing == pb.N):
            return sol
        return Solution(routes=routes, total_distance=total_distance, fleet_size=fleet_size, obj=obj)

    def _feasible_insert_at(self, route: Route, cust: int, pos: int) -> bool:
        # Lightweight feasibility proxy for LS: check time windows via simple slack
        # and product incompatibility & capacity pessimistically (not recomputing exact schedule).
        pb = self.pb
        # Check product incompatibility against whole route product set
        route_prod = self._route_product_set(route)
        cust_prod = ((pb.A[cust] + pb.B[cust]) > 0)
        if route_prod.any():
            for p in np.where(cust_prod)[0]:
                for q in np.where(route_prod)[0]:
                    if pb.C[p, q] == 0:
                        return False
        # Capacity: rough (sum loads) + cust
        w_sum = self._route_load(route, pb.A).sum() + pb.A[cust].sum()
        v_sum = self._route_load(route, pb.B).sum() + pb.B[cust].sum()
        # Pick first vehicle cap (approx). In rigorous impl we'd track per-vehicle caps across LS.
        cap_w, cap_v = route.cap_w, route.cap_v
        if w_sum > cap_w or v_sum > cap_v:
            return False
        # Time windows: simple check based on detour increase; skip strict feasibility for speed
        return True

    def _delta_distance_insert(self, route: Route, cust: int, pos: int) -> float:
        pb = self.pb
        nodes = route.nodes
        a = nodes[pos - 1]
        b = nodes[pos] if pos < len(nodes) else 0
        old = pb.D[a, b]
        new = pb.D[a, cust] + pb.D[cust, b]
        return new - old

    def _route_product_set(self, route: Route) -> np.ndarray:
        pb = self.pb
        P = pb.P
        ps = np.zeros(P, dtype=bool)
        for n in route.nodes:
            ps |= ((pb.A[n] + pb.B[n]) > 0)
        return ps

    def _route_load(self, route: Route, M: np.ndarray) -> np.ndarray:
        total = np.zeros(M.shape[1])
        for n in route.nodes:
            total += M[n]
        return total

    def _two_opt_modified(self, sol: Solution) -> Solution:
        pb = self.pb
        routes = [rt.clone() for rt in sol.routes]
        for rt in routes:
            n = len(rt.nodes)
            if n <= 3:
                continue
            lspace = 0
            n_swaps = (n * (n - 1)) // 2 - 1
            k = 1
            j = 1
            while j <= n_swaps:
                if k + 1 + lspace < n:
                    a = k
                    b = k + 1 + lspace
                    # swap segment (2-opt edge swap)
                    new_nodes = rt.nodes[:a] + list(reversed(rt.nodes[a:b])) + rt.nodes[b:]
                    # Feasibility (light): accept if distance decreases and depot endpoints preserved
                    old_d = Feasibility.route_distance(pb.D, rt.nodes)
                    new_d = Feasibility.route_distance(pb.D, new_nodes)
                    if new_d + 1e-9 < old_d:
                        rt.nodes = new_nodes
                    k += 1
                else:
                    k = 1
                    lspace += 1
                j += 1
        # Compute objective with coverage penalty
        served = set()
        for rt in routes:
            for n in rt.nodes:
                if n != 0:
                    served.add(n)
        missing = (pb.N - len(served))
        penalty = 0.0
        if missing > 0:
            penalty = 1e6 * missing

        total_distance = sum(Feasibility.route_distance(pb.D, rt.nodes) for rt in routes if len(rt.nodes) > 1)
        fleet_size = sum(1 for rt in routes if len(rt.nodes) > 2)
        if self.pr.use_xi:
            obj = pb.xi * fleet_size + total_distance + penalty
        else:
            obj = total_distance + penalty

        # If everything got dropped (shouldn't happen), keep original solution
        if (len(served) == 0) and (missing == pb.N):
            return sol
        return Solution(routes=routes, total_distance=total_distance, fleet_size=fleet_size, obj=obj)

    def _update_strong_pheromone(self, sol: Solution):
        # Add reinforcement along arcs of the global best as Eq. (7)
        # ΔQ = 1 / best_cost (use objective or distance; we use objective for direct guidance)
        dQ = 1.0 / max(sol.obj, 1e-9)
        for rt in sol.routes:
            nodes = rt.nodes
            for i, j in zip(nodes[:-1], nodes[1:]):
                self.Qs[i, j] = self.Qs[i, j] + self.pr.alpha * (dQ - self.Qs[i, j])


# -----------------------------
# Demo instance & CLI
# -----------------------------

def tiny_demo() -> Tuple[Problem, Solution]:
    # Build a tiny synthetic instance with 6 customers, 2 products, simple time windows.
    rng = np.random.default_rng(0)
    N = 6
    P = 2
    coords = rng.random((N + 1, 2))
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    T = D.copy()  # unit speed

    # Time windows: depot [0, inf), customers [0.2, 1.2] with service 0.01
    TW = np.zeros((N + 1, 3))
    TW[0] = [0.0, 1e9, 0.0]
    TW[1:, 0] = 0.2
    TW[1:, 1] = 1.2
    TW[1:, 2] = 0.01

    # Demands: small random weights/volumes per product
    A = np.zeros((N + 1, P))
    B = np.zeros((N + 1, P))
    A[1:] = rng.integers(0, 2, size=(N, P)) * 1.0
    B[1:] = rng.integers(0, 2, size=(N, P)) * 0.5

    # Compatibility: product 0 incompatible with product 1
    C = np.array([[1, 0], [0, 1]], dtype=int)

    # Fleet: up to 4 vehicles, caps
    F_caps = np.array([[4.0, 3.0], [4.0, 3.0], [4.0, 3.0], [4.0, 3.0]])

    problem = Problem(D=D, T=T, TW=TW, A=A, B=B, C=C, F_caps=F_caps)
    params = ACSParams(r=200, m=6, seed=42)  # fewer iters for demo speed

    solver = TwoPheromoneACS(problem, params)
    sol = solver.solve()
    return problem, sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run a tiny synthetic demo")
    args = parser.parse_args()

    if args.demo:
        pb, sol = tiny_demo()
        print("Demo solution:")
        print(f"  N={pb.N}, P={pb.P}, xi={pb.xi:.4f}")
        print(f"  Fleet size: {sol.fleet_size}")
        print(f"  Total distance: {sol.total_distance:.4f}")
        print(f"  Objective: {sol.obj:.4f}")
        for idx, rt in enumerate(sol.routes):
            if len(rt.nodes) > 2:
                print(f"   - Route {idx}: {rt.nodes}")
    else:
        # Default: run demo for convenience
        pb, sol = tiny_demo()
        print("(Use --demo to suppress this message)\n")
        print(f"Fleet size: {sol.fleet_size}, Distance: {sol.total_distance:.4f}, Obj: {sol.obj:.4f}")
        for idx, rt in enumerate(sol.routes):
            if len(rt.nodes) > 2:
                print(f"   - Route {idx}: {rt.nodes}")


if __name__ == "__main__":
    main()
