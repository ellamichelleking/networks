import numpy as np

'''
Lefthand side of an ODE describing adaptiation in a network, adopted from 
Phenotypes of Vascular Flow Networks, Ronellenfitsch and Katifori, 2019

Inputs:
    - K: dimensionless conductivity
    - t: time
    - netw: network (nodes and edges)
    - currents_function: functon that describes how to update the currents 
                         in the network given the state of the network
    - kappa: dimensionless background strength
    - beta: nonlinearity
    - rho: decay timescale
'''
def adaptation_ode(K, t, netw, currents_function, kappa, beta, rho):
    F = currents_function(K, netw)
    return F ** beta - K + kappa * np.exp(-t / rho)



'''
Steady state solver using Euler method

Inputs:
    - f: function to find steady state solution for
    - x0: initial parameter values
    
Returns:
    - parameter values that achieve steady state
    - boolean that describes whether the solver converged

'''
def ss_solve(f, x0, atol=1e-8, rtol=1e-6, Δt=0.1, maxint=100000):
    x = np.copy(x0)
    fx = f(x, 0.0)

    converged = False
    for i in range(1,maxint+1):
        t = i * Δt
        fx_new = f(x, t)

        Δfx = np.linalg.norm(fx - fx_new)
        if (Δfx < rtol * np.linalg.norm(fx)) or (Δfx < atol):
            converged = True
            break

        fx = fx_new

        # time step
        x = x + Δt * fx
        
    return x, converged
