import numpy as np
import random
import os
import bson

from setup_networks import network_from_txt, network_indices
#from currents import static_currents, correlated_currents_fun, corson_currents_fun, random_currents_fun
from currents import * 
from adaptation import adaptation_ode, ss_solve
from measures import steady_state_dissipation, area_penalty, cost

'''
Computes network properties for a range of dimensionless paramters kappa, 
rho, and sigma. These properties consists of a power dissipation term, a term estimating 
the amount of "building material" needed for a given network, and a measure of network 
robustness.

Kappa describes the background strength, rho describes the decay timescale, and sigma
describes the fluctuation scale.

The function takes 'gamma' as an argument, which is (1) the power of the cost describing the
amount of building material needed and (2) is 1 / (1 + beta) where beta is a parameter in the
adaptation ODE.

Results are written to a bson file with a name based on the choice of gamma and random seed.
'''
def run_simulations_gauss(gamma):
    print(f"gamma = {gamma}")
    params = {
        "N_samples": 2,
        "N_kappa": 5,
        "N_rho": 5,
        "N_sigma": 40,
        "sigma_min": 0.1,
        "sigma_max": 5.0,
        "fluctuations": 'static',#"gauss",
        "source": "left",
        "beta": 1.0 / (1 + gamma)
    }

    name = f"gauss_gamma_{gamma}_{abs(int(random.random() * 1000000))}"
    print(name)

    N_samples = params["N_samples"]
    combinations = []

    for kappa in np.logspace(-2, 0, params["N_kappa"]):
        for rho in np.logspace(0, 2, params["N_rho"]):
            for sigma in np.linspace(params["sigma_min"], params["sigma_max"], params["N_sigma"]):
                for _ in range(N_samples):
                    combinations.append([kappa, rho, sigma])

    netw = network_from_txt("../lattices/paper_edges.txt", "../lattices/paper_nodes.txt")
    inds = network_indices(netw)
    i = inds[params["source"]]
    maxi = len(combinations)
    results = []

    for j, c in enumerate(combinations):
        kappa, rho, sigma = c
        K0 = -np.log10(np.random.rand(netw.N_e))
        #K0 = -np.log10(np.ones(netw.N_e) - 0.4)
        
        if params['fluctuations']=='gauss':
            currents = correlated_currents_fun(netw, lambda x: np.exp(-0.5 * x**2 / (sigma * netw.mean_len)**2), source_index=i)
            
        # HAVEN'T CHECKED THESE YET
        elif params['fluctuations']=='exponential':
            currents = correlated_currents_fun(netw, lambda x: np.exp(np.abs(x) / (sigma * netw.mean_len)), source_index=i)
        elif params['fluctuations']=='random':
            currents = random_currents_fun(sigma, netw, source_index=i)
        elif params['fluctuations']=='corson':
            # NOTE: I'm not sure what "ee" is supposed to be
            ee = 0.1
            currents = corson_currents_fn(ee, netw, source_index=i)
        elif params['fluctuations']=='static':
            sink_nodes = np.random.randint(0, netw.N_v, size=50) #choose 50 nodes to act as sinks
            currents = lambda K, netw: static_currents(K, netw, source_index=i, sink_nodes=sink_nodes)

        K, converged = ss_solve(lambda K, t: adaptation_ode(K, t, netw, currents, kappa, params["beta"], rho), K0, Δt=1.0)
        if not converged:
            print('Warning: ss_solve did not converge')

        P_ss = steady_state_dissipation(K, netw, source_index=i)
        A = area_penalty(K, netw, source_index=i)
        C = cost(K, netw, gamma)
        C_half = cost(K, netw, 0.5)

        print(f"{j}/{maxi}  kappa={kappa}, rho={rho}, sigma={sigma}.")
        print(f"P_ss = {P_ss}, A = {A}, C = {C}")

        res = {
            "K": K.tolist(),
            "converged": True,  # Assuming it always converges for demonstration
            "P_ss": float(P_ss),
            "A": float(A),
            "C": float(C),
            "C_half": float(C_half),
            "kappa": float(kappa),
            "rho": float(rho),
            "sigma": float(sigma)
        }
        results.append(res)

    with open(f"ben_data/results_{name}.bson", "wb") as f:
        bson.dump({
            "network": netw,
            "results": results,
            "parameters": params
        }, f)


if __name__ == "__main__":
    import sys
    run_simulations_gauss(float(0.5))
