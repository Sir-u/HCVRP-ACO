# aco_hcvrp_min_sum.py  (ACS for HCVRP / min-sum)
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np, math, random, time

@dataclass
class ACOParams:
    # m (ants)
    ants: int = 20
    # δ, β (exponents on pheromone and heuristic, respectively)
    alpha: float = 3.0   # <- δ
    beta: float = 3.0    # <- β
    # α (global evap), γ (local decay)
    rho: float = 0.10    # <- α
    xi: float = 0.10     # <- γ
    # ω0 (we map to ACS q0: exploitation prob; in the cited table ω0=0.2)
    q0: float = 0.20
    # k-ratio for candidate restriction (k ≈ W ≈ 0.4N in user's model)
    k_ratio: float = 0.4
    two_opt: bool = True
    cap_select: str = "max"  # "max" or "min_feasible"
    objective: str = "distance"  # "distance" or "time"
    v_speeds: Optional[List[float]] = None  # Optional explicit speeds per capacity rank (e.g., V3: [1/4,1/5,1/6])
    # stopping
    iters: int = 200
    time_budget_s: Optional[float] = None
    seed: Optional[int] = 24610

def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a-b))

def _route_length(route: List[int], D: np.ndarray) -> float:
    L = 0.0
    prev = 0
    for v in route:
        L += float(D[prev, v])
        prev = v
    L += float(D[prev, 0])
    return L

def _two_opt(route: List[int], D: np.ndarray) -> List[int]:
    improved = True
    best = route[:]
    best_len = _route_length(best, D)
    n = len(best)
    if n < 4:
        return best
    while improved:
        improved = False
        for i in range(n-3):
            for j in range(i+2, n-1):
                new_route = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                L = _route_length(new_route, D)
                if L + 1e-12 < best_len:
                    best = new_route
                    best_len = L
                    improved = True
                    break
            if improved:
                break
    return best

def _tau0_Q0(coords_all: np.ndarray) -> float:
    """Q0 = (N-1) / sum_{i≠j} D_ij  with nodes indexed [0..N] (0 is depot)."""
    nodes = coords_all.shape[0]        # N+1 nodes including depot
    N_minus_1 = nodes - 1              # customers count (N)
    Dsum = 0.0
    for i in range(nodes):
        for j in range(nodes):
            if i == j: 
                continue
            Dsum += float(np.linalg.norm(coords_all[i] - coords_all[j]))
    return (N_minus_1 / (Dsum + 1e-12))

def _candidate_lists(D: np.ndarray, k: int):
    Np1 = D.shape[0]
    cand = [[] for _ in range(Np1)]
    for i in range(Np1):
        order = np.argsort(D[i])
        cand[i] = [int(j) for j in order if j != i][:k]
    return cand

def _choose_capacity(caps: List[float], mode: str, feasible_demands: List[float]) -> float:
    if mode == "max":
        return float(max(caps))
    elif mode == "min_feasible":
        sorted_caps = sorted(caps)
        need = max(feasible_demands) if feasible_demands else min(sorted_caps)
        for c in sorted_caps:
            if c >= need:
                return float(c)
        return float(sorted_caps[-1])
    else:
        return float(max(caps))

def _speed_from_capacity(cap: float, caps: List[float], v_speeds: Optional[List[float]] = None) -> float:
    """
    Speed model for MS-HCVRP (min-sum time).
    - If v_speeds is provided, map capacity rank -> fixed speed (e.g., V3: [1/4,1/5,1/6]).
    - Otherwise fall back to inverse-capacity normalization (cmin/c).
    """
    if v_speeds is not None:
        uniq = sorted(set(float(c) for c in caps))
        c = float(cap)
        assert c in uniq, "Capacity not found in caps set"
        idx = uniq.index(c)
        assert idx < len(v_speeds), "v_speeds length must match number of unique capacities"
        return float(v_speeds[idx])
    # Fallback: inverse-capacity normalized by smallest capacity
    cmin = float(min(caps))
    return cmin / float(cap + 1e-12)

def solve_instance(
    depot_xy: np.ndarray, customers_xy: np.ndarray, demand: np.ndarray, caps: List[float],
    params: ACOParams = ACOParams()
) -> Tuple[float, List[List[int]]]:
    if params.seed is not None:
        random.seed(params.seed); np.random.seed(params.seed)

    N = customers_xy.shape[0]
    coords_all = np.vstack([depot_xy.reshape(1,2), customers_xy])

    # Precompute all pairwise distances once (speeds up 2-opt and routing)
    D = np.sqrt(((coords_all[:,None,:] - coords_all[None,:,:])**2).sum(-1))

    # Heuristic & Pheromone
    eta = 1.0 / (D + 1e-12)
    np.fill_diagonal(eta, 0.0)
    tau0 = _tau0_Q0(coords_all)  # ← from the table (Q0)
    tau = np.full((N+1, N+1), tau0, dtype=float)
    np.fill_diagonal(tau, 0.0)

    k = max(1, int(round(params.k_ratio * N)))
    cand = _candidate_lists(D, k=k)

    best_cost = float("inf"); best_solution: List[List[int]] = []
    t0 = time.time()
    def time_up():
        return (params.time_budget_s is not None) and (time.time() - t0 >= params.time_budget_s)

    for it in range(params.iters):
        iter_best_cost = float("inf"); iter_best_sol: List[List[int]] = []
        for m in range(params.ants):
            unvisited = set(range(1, N+1))
            routes: List[List[int]] = []
            route_speeds: List[float] = []
            while unvisited:
                feas_demands = [float(demand[j-1]) for j in unvisited]
                cap = _choose_capacity(list(caps), params.cap_select, feas_demands)
                speed = _speed_from_capacity(cap, list(caps), params.v_speeds)
                rem = cap; cur = 0; route: List[int] = []
                while True:
                    feas = [j for j in unvisited if demand[j-1] <= rem + 1e-12]
                    if not feas: break
                    neigh = [j for j in cand[cur] if j in feas]
                    C = neigh if neigh else feas
                    if not C: break
                    # ACS pseudo-random proportional rule
                    if random.random() < params.q0:   # exploit
                        nxt = max(C, key=lambda j: (tau[cur,j]**params.alpha) * (eta[cur,j]**params.beta))
                    else:                              # explore
                        w = np.array([(tau[cur,j]**params.alpha) * (eta[cur,j]**params.beta) for j in C], dtype=float)
                        s = float(w.sum())
                        if s <= 0: nxt = random.choice(C)
                        else:
                            r = random.random() * s; acc = 0.0; nxt = C[-1]
                            for j, ww in zip(C, w):
                                acc += float(ww)
                                if r <= acc:
                                    nxt = j; break
                    route.append(nxt); unvisited.remove(nxt); rem -= float(demand[nxt-1])
                    # local update (γ)
                    tau[cur, nxt] = (1.0 - params.xi) * tau[cur, nxt] + params.xi * tau0
                    tau[nxt, cur] = tau[cur, nxt]
                    cur = nxt
                    if not [j for j in unvisited if demand[j-1] <= rem + 1e-12]: break
                if len(route) > 1 and params.two_opt:
                    route = _two_opt(route, D)
                routes.append(route)
                route_speeds.append(speed)
            if params.objective == "time":
                cost = sum(_route_length(r, D) / s for r, s in zip(routes, route_speeds))
            else:
                cost = sum(_route_length(r, D) for r in routes)
            if cost < iter_best_cost:
                iter_best_cost = cost; iter_best_sol = routes
        # global evaporation (α) and reinforcement
        if iter_best_cost < best_cost:
            best_cost = iter_best_cost; best_solution = [r[:] for r in iter_best_sol]
        tau *= (1.0 - params.rho)
        delta = 1.0 / (iter_best_cost + 1e-12)
        used = set()
        for r in iter_best_sol:
            prev = 0
            for v in r:
                used.add((prev, v)); prev = v
            used.add((prev, 0))
        for (i, j) in used:
            tau[i,j] += params.rho * delta
            tau[j,i] = tau[i,j]
        if time_up():
            break
    return best_cost, best_solution
