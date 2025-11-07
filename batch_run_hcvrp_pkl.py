#!/usr/bin/env python3
"""
Batch runner for HCVRP instances stored in a PKL.
- Expects the PKL to contain an ndarray of shape (K, 4) where each row is:
  [depot_coord, customer_coords_list, demand_list, vehicle_caps_list].
- Converts each instance to the Problem schema used by TwoPheromoneACS and runs the solver.
- Writes a CSV with results and a JSONL with per-instance routes.

Usage:
  python batch_run_hcvrp_pkl.py --pkl /path/to/hcvrp_40_seed24610.pkl --out results.csv --routes routes.jsonl \
      --iters 1500 --agents 10 --seed 0 --start 0 --end 1280

You must have `two_pheromone_acs_hvrptwmpic.py` importable (same directory or PYTHONPATH).
"""
import argparse, csv, json, time, pickle, os, sys, math
from typing import List
import numpy as np

# Import solver
try:
    from two_pheromone_acs_hvrptwmpic import Problem, ACSParams, TwoPheromoneACS
except Exception as e:
    print("ERROR: Could not import two_pheromone_acs_hvrptwmpic. Make sure the file is available.", file=sys.stderr)
    raise

def build_problem_from_instance(inst_row):
    depot, customers, demand, caps = inst_row
    customers = np.asarray(customers, dtype=float)
    depot = np.asarray(depot, dtype=float)
    coords = np.vstack([depot[None, :], customers])  # (N+1, 2)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    T = D.copy()

    Np1 = coords.shape[0]
    TW = np.zeros((Np1, 3), dtype=float)
    TW[0] = [0.0, 1e9, 0.0]       # depot
    TW[1:, 0] = 0.0               # earliest
    TW[1:, 1] = 1e9               # latest
    TW[1:, 2] = 0.0               # service time

    demand = np.asarray(demand, dtype=float).reshape(-1)
    A = np.zeros((Np1, 1), dtype=float)
    A[1:, 0] = demand             # single-product weight demand
    B = np.zeros_like(A)          # no volume demand
    C = np.ones((1, 1), dtype=int)

    caps = np.asarray(caps, dtype=float).reshape(-1)
    # Map to (weight, volume). We mirror the same capacity in both dims to keep constraints active.
    F_caps = np.stack([caps, caps], axis=1)

    return Problem(D=D, T=T, TW=TW, A=A, B=B, C=C, F_caps=F_caps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="Path to PKL file (numpy ndarray of K instances).")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--routes", required=True, help="Output JSONL path (routes per instance).")
    ap.add_argument("--iters", type=int, default=1500, help="ACS iterations per instance (r).")
    ap.add_argument("--agents", type=int, default=10, help="Number of ants (m).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--start", type=int, default=0, help="Start index (inclusive).")
    ap.add_argument("--end", type=int, default=-1, help="End index (exclusive). -1 = all.")
    ap.add_argument("--every", type=int, default=1, help="Stride (process every k-th instance).")
    ap.add_argument("--no_xi", action="store_true", help="Use distance-only objective (min-sum)")
    ap.add_argument("--multitrip", action="store_true", help="Allow multiple trips per vehicle (K fixed)")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("PKL must be a numpy ndarray of shape (K, 4). Got shape: %r" % (getattr(data, "shape", None),))

    K = data.shape[0]
    start = max(0, args.start)
    end = K if args.end < 0 else min(args.end, K)

    # Prepare outputs
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.routes) or ".", exist_ok=True)
    csv_exists = os.path.isfile(args.out)
    csv_f = open(args.out, "a", newline="")
    writer = csv.writer(csv_f)
    if not csv_exists:
        writer.writerow(["idx", "N", "fleet", "distance", "objective", "xi", "iters", "agents", "seed", "elapsed_sec"])

    routes_f = open(args.routes, "a")

    total_start = time.time()
    for idx in range(start, end, args.every):
        inst = data[idx]
        pb = build_problem_from_instance(inst)
        params = ACSParams(r=args.iters, m=args.agents, seed=args.seed, use_xi=(not args.no_xi), allow_multitrip=args.multitrip)
        t0 = time.time()
        sol = TwoPheromoneACS(pb, params).solve()
        t1 = time.time()

        # Count distinct vehicles actually used
        used_vehicles = {rt.vehicle_idx for rt in sol.routes if len(rt.nodes) > 2 and rt.vehicle_idx >= 0}
        fleet_for_csv = len(used_vehicles)

        # CSV line
        writer.writerow([idx, pb.N, fleet_for_csv, f"{sol.total_distance:.6f}", f"{sol.obj:.6f}",
                         f"{pb.xi:.6f}", params.r, params.m, params.seed, f"{t1 - t0:.3f}"])
        csv_f.flush()

        # JSONL routes
        routes_obj = {
            "idx": idx,
            "routes": [rt.nodes for rt in sol.routes if len(rt.nodes) > 2],
            "fleet": fleet_for_csv,
            "distance": sol.total_distance,
            "objective": sol.obj
        }
        routes_f.write(json.dumps(routes_obj) + "\n")
        routes_f.flush()

        # Progress
        print(f"[{idx}/{end}) N={pb.N} fleet={fleet_for_csv} dist={sol.total_distance:.4f} obj={sol.obj:.4f} time={t1-t0:.2f}s")

    csv_f.close()
    routes_f.close()
    print(f"Done. Processed {len(range(start, end, args.every))} instances in {time.time() - total_start:.1f}s.")

if __name__ == "__main__":
    main()
