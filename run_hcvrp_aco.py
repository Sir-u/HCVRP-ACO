import pickle, numpy as np
from two_pheromone_acs_hvrptwmpic import Problem, ACSParams, TwoPheromoneACS
import argparse

def build_problem(inst):
    depot, customers, demand, caps = inst
    customers = np.asarray(customers, float)
    depot = np.asarray(depot, float)
    coords = np.vstack([depot[None, :], customers])
    D = np.linalg.norm(coords[:,None,:]-coords[None,:,:], axis=2)
    T = D.copy()
    Np1 = coords.shape[0]
    TW = np.zeros((Np1,3), float)
    TW[0] = [0.0, 1e9, 0.0]
    demand = np.asarray(demand, float).reshape(-1)
    A = np.zeros((Np1,1), float); A[1:,0] = demand
    B = np.zeros_like(A)
    C = np.ones((1,1), int)
    caps = np.asarray(caps, float).reshape(-1)
    F_caps = np.stack([caps, caps], axis=1)
    return Problem(D=D, T=T, TW=TW, A=A, B=B, C=C, F_caps=F_caps)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", default="data/hcvrp/hcvrp_40_seed24610.pkl")
    p.add_argument("--idx", type=int, default=1000)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--agents", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_xi", action="store_true", help="Use distance-only objective (min-sum)")
    p.add_argument("--multitrip", action="store_true", help="Allow multiple trips per vehicle (K fixed)")
    args = p.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    pb = build_problem(data[args.idx])
    params = ACSParams(r=args.iters, m=args.agents, seed=args.seed, use_xi=(not args.no_xi), allow_multitrip=args.multitrip)
    sol = TwoPheromoneACS(pb, params).solve()
    print(f"Fleet(vehicles used)={sol.fleet_size}, Dist={sol.total_distance:.3f}, Obj={sol.obj:.3f}, use_xi={not args.no_xi}, multitrip={args.multitrip}")
    from two_pheromone_acs_hvrptwmpic import Feasibility
    for i, rt in enumerate(sol.routes):
        if len(rt.nodes) > 2:
            d = Feasibility.route_distance(pb.D, rt.nodes)
            print(f"Route {i} (veh {rt.vehicle_idx}): dist={d:.3f}, nodes={rt.nodes}")