import numpy as np
from currents import static_currents

'''
Hydraulic power dissipation due to steady state flows (nonfluctuating conditions)

Arguments:
    - K: conductivities
    - netw: network geometry, stored in a Network object
'''
def steady_state_dissipation(K, netw, source_index=0):
    F2 = static_currents(K, netw, source_index=source_index)
    nonz = K > 1e-8
    return np.sum(netw.lengths[nonz] * F2[nonz] / K[nonz])



'''
Measure of network robustness, computed by looking at a percolation penalty:
the expected fraction of perfused area lost when removing an edge 

Arguments:
    - K: conductivities
    - netw: network geometry, stored in a Network object
    - source_index: index of node acting as source in the system
'''
def area_penalty(K, netw, source_index=0):
    K_copy = np.copy(K)
    K_copy[K_copy < 1e-7] = 0.0

    E = netw.E[1:, :]
    L = E @ np.diag(K_copy) @ E.T #Laplacian

    q = -np.ones(netw.N_v) #uniform sinks
    q[source_index] = netw.N_v - 1

    q_reduced = q[1:]

    p = np.linalg.solve(L, q_reduced)
    F = np.abs(K_copy * (E.T @ p)) #compute flows in the usual way
    
    S = E.multiply(np.linalg.inv(L) @ E)

    # NOTE: this matches what happens in the julia code, but it seems very weird to throw out all the S data
    s = K_copy * np.sum(S, axis=1)[0,0]

    divergent = s > 1 - 1e-5

    penalty = np.mean(F[divergent]) / netw.N_v if np.any(divergent) else 0.0
    return penalty


'''
Cost of building the network, computed by estimating 
the amount of material requried to build the network.
Longer edges and higher conductivities (wider eges) 
are treated as using more material.

Arguments:
    - K: conductivities
    - netw: network geometry, stored in a Network object
    - gamma: gamma < 1 indicates an economy of scale 

'''
def cost(K, netw, gamma=0.5):
    return np.sum(netw.lengths * K**gamma)
