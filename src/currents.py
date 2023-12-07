import numpy as np
from scipy.linalg import solve
import random


'''
Computes the flows in the network assuming currents are defined by one
source specified by source_index and with all other nodes acting as uniform sinks

Arguments:
    - K: conductivities
    - netw: network geometry, stored in a Network object
    - source_index: index of node acting as source in the system
    - sink_nodes: indices of nodes to act as sinks. If None, all non-source nodes are sinks
    - eps_current: value of tiny sink to give passive nodes to prevent numerical instabilities
Returns:
    - flows squared
'''
def static_currents(K, netw, source_index=0, sink_nodes=None, eps_sink = 5e-4):
    E = netw.E[1:, :] # E is the indcidence matrix
    L = E @ np.diag(K) @ E.T # L is the Laplacian
    if sink_nodes is None:
        q = -np.ones(netw.N_v) / (netw.N_v - 1) # assumes everything but the source is a sink
    else:
        q = -np.ones(netw.N_v) * eps_sink
        q[sink_nodes] = -1.0
        q[source_index] = 0
        q /= np.sum(q)
    q[source_index] = 1.0
    q_reduced = q[1:]

    p = np.linalg.solve(L, q_reduced) #p is the pressure
    F = K * (E.T @ p) # flows = conductivities * pressures

    return F ** 2


'''
Computes the flows in the network assuming correlations in the currents in 
the network that are described by function f, which acts on distances between nodes

Arguments:
    - netw: network geometry, stored in a Network object
    - f: function describing decays as a function of distance
Returns:
    - function to compute the currents with conductivities K and network structure netw as arguments

'''
def correlated_currents_fun(netw, f, source_index=0):
    pos = netw.pos
    D = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2) # matrix of distances between nodes
    Q = -f(D)

    # Normalize the currents: everything but the source should sum to -1
    Q[source_index, :] = 0.0
    sums = np.sum(Q, axis=0)   
    Q /= -sums[np.newaxis, :]
    Q[source_index, :] = 1.0

    Q_reduced = Q[1:]    
    
    def currents_fun(K, netw):
        E = netw.E[1:, :]
        L = E @ np.diag(K) @ E.T

        p = np.linalg.solve(L, Q_reduced)
        F = K[:, np.newaxis] * (E.T @ p)
        return np.mean(F ** 2, axis=1)

    return currents_fun



'''
Computes the flows in the network assuming correlations in the currents in 
the network according to those described by Corson in `Fluctuations and Redundancy 
in Optimal Transport Networks'

Arguments:
    - netw: network geometry, stored in a Network object
    - f: function describing decays as a function of distance
Returns:
    - function to compute the currents with conductivities K and network structure netw as arguments

'''
def corson_currents_fun(ee, netw, source_index=0):
    Q = np.zeros((netw.N_v, netw.N_v))

    Q -= (1.0 - ee)
    Q[np.arange(netw.N_v), np.arange(netw.N_v)] -= ee
    Q = np.delete(Q, source_index, axis=0)

    Q[source_index] = 0.0
    sums = np.sum(Q, axis=0)
    Q /= -sums[:, np.newaxis]
    Q[source_index] = 1.0

    Q_reduced = Q[1:]

    def currents_fun(K, netw):
        E = netw.E[1:, :]
        L = E @ np.diag(K) @ E.T

        p = np.linalg.solve(L, Q_reduced)
        F = K[:, np.newaxis] * (E.T @ p)

        return np.mean(F ** 2, axis=1)

    return currents_fun




'''
Computes the flows in the network assuming currents in the network are initialized randomly

Arguments:
    - p: probability of being a sink versus a passive node
    - netw: network geometry, stored in a Network object
    - source_index: index of node acting as source in the system
Returns:
    - function to compute the currents with conductivities K and network structure netw as arguments

'''
def random_currents_fun(p, netw, source_index=0):
    Q = np.random.choice([1.0, 0.0], size=(netw.N_v, netw.N_v), p=[p, 1 - p])

    Q[source_index] = 0.0
    sums = np.sum(Q, axis=0)
    Q /= -sums[:, np.newaxis]
    Q[source_index] = 1.0

    Q_reduced = Q[1:]

    def currents_fun(K, netw):
        E = netw.E[1:, :]
        L = E @ np.diag(K) @ E.T

        p = np.linalg.solve(L, Q_reduced)
        F = K[:, np.newaxis] * (E.T @ p)
        return np.mean(F ** 2, axis=1)

    return currents_fun


