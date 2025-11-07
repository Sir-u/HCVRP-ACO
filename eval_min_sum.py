import pickle, os, csv, time, math, numpy as np
from typing import Optional
from aco_hcvrp_min_sum import solve_instance, ACOParams

# eval_min_sum.py  (iteration schedule from the provided table)
def _parse_v_speeds(spec: str) -> Optional[list]:
    if spec is None:
        return None
    s = spec.strip()
    if not s:
        return None
    u = s.upper()
    if u == "V3":
        return [1/4, 1/5, 1/6]
    if u == "V5":
        return [1/4, 1/5, 1/6, 1/7, 1/8]
    # comma-separated floats
    return [float(x) for x in s.split(",")]

def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def iter_for_size(N: int) -> int:
    # Table: it(40)=1e4, it(60)=1.2e4, ..., it(160)=2.2e4
    key = [40, 60, 80, 100, 120, 140, 160]
    val = [10000, 12000, 14000, 16000, 18000, 20000, 22000]
    if N in key:
        return val[key.index(N)]
    # linear interpolate/extrapolate on 20-step grid
    if N < key[0]:
        k0, v0 = key[0], val[0]
        slope = (val[1]-val[0])/(key[1]-key[0])
        return max(1000, int(round(v0 + slope*(N-k0))))
    if N > key[-1]:
        k0, v0 = key[-1-1], val[-1-1]
        k1, v1 = key[-1],   val[-1]
        slope = (v1-v0)/(k1-k0)
        return int(round(v1 + slope*(N-k1)))
    for i in range(len(key)-1):
        if key[i] < N < key[i+1]:
            k0, v0 = key[i], val[i]
            k1, v1 = key[i+1], val[i+1]
            t = (N-k0)/(k1-k0)
            return int(round(v0 + t*(v1-v0)))
    return 10000

def main(dataset_path: str, out_csv: str, time_budget_s: float = None,
         idx: int = None, ants: int = 20, iters_override: int = None,
         two_opt: bool = True, k_ratio: float = 0.4, cap_select: str = "max",
         objective: str = "distance", v_speeds: str = None):
    data = load_dataset(dataset_path)
    M = data.shape[0]
    N = len(data[0,2])
    iters = iter_for_size(N)
    if iters_override is not None:
        iters = iters_override
    indices = [idx] if idx is not None else range(M)
    parsed_v = _parse_v_speeds(v_speeds)
    params = ACOParams(iters=iters, time_budget_s=time_budget_s,
                       ants=ants, q0=0.20, rho=0.10, xi=0.10, alpha=3.0, beta=3.0,
                       two_opt=two_opt, k_ratio=k_ratio, cap_select=cap_select,
                       objective=objective, v_speeds=parsed_v)
    print(f"Running ACO on {len(indices)} instance(s) (N={N}) out of {M} total with params: {params}")
    t0 = time.time()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        # elapsed_s 컬럼 추가
        w.writerow(["idx","N","iters","cost_min_sum","num_routes","len_routes","objective","v_speeds","elapsed_s"])
        for i in indices:
            depot = np.array(data[i,0], dtype=float)
            custs = np.array(data[i,1], dtype=float)
            demand = np.array(data[i,2], dtype=float)
            caps = list(map(float, data[i,3]))
            t_i = time.time()
            cost, routes = solve_instance(depot, custs, demand, caps, params=params)
            elapsed = time.time() - t_i
            vcol = "" if params.v_speeds is None else ";".join(f"{v:.6g}" for v in params.v_speeds)
            w.writerow([i, len(demand), params.iters, f"{cost:.6f}", len(routes),
                        ";".join(str(len(r)) for r in routes), params.objective, vcol, f"{elapsed:.6f}"])
            if (i+1) % 50 == 0:
                print(f"  {i+1}/{M} done")
    print(f"Done in {time.time()-t0:.2f}s. Results -> {out_csv}")

if __name__ == "__main__":
    import argparse, numpy as np
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--out", type=str, default="aco_results.csv")
    p.add_argument("--time_budget_s", type=float, default=None)
    p.add_argument("--idx", type=int, default=None, help="If provided, only run this instance index")
    p.add_argument("--ants", type=int, default=20)
    p.add_argument("--iters", type=int, default=None, help="If provided, override iteration schedule")
    p.add_argument("--k_ratio", type=float, default=0.4)
    p.add_argument("--cap_select", type=str, default="max", choices=["max","min_feasible"])
    p.add_argument("--objective", type=str, default="distance", choices=["distance","time"])
    p.add_argument("--v_speeds", type=str, default=None, help="Comma-separated speeds, or 'V3'/'V5' preset")
    p.add_argument("--no_two_opt", action="store_true")
    args = p.parse_args()
main(args.dataset, args.out, args.time_budget_s,
     idx=args.idx, ants=args.ants, iters_override=args.iters,
     two_opt=(not args.no_two_opt), k_ratio=args.k_ratio, cap_select=args.cap_select,
     objective=args.objective, v_speeds=args.v_speeds)
