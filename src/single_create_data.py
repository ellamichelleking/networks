import numpy as np
import random
import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from tqdm import tqdm

from setup_networks import network_from_txt, network_indices, Network, network_from_edges_and_nodes, triangular_lattice_pts, get_edges, save_network
from currents import static_currents
from adaptation import adaptation_ode, ss_solve
from measures import steady_state_dissipation
from phase_diagrams import make_ellipse_netw, get_sinks


parser = argparse.ArgumentParser(description='Create data for placenta network analysis')
parser.add_argument('-f', '--filename', type=str, default="expgrowth")
parser.add_argument('-el', '--ellipse_ratio', type=float, default=1.0, help='ratio of ellipse axis lengths')
parser.add_argument('-N', '--N_nodes', type=int, default=10000)
parser.add_argument('-len', '--edge_len', type=float, default=0.08) 
#parser.add_argument('-Ns', '--N_sinks', type=int, default=90)
parser.add_argument('-g', '--gamma', type=float, default=0.5)
parser.add_argument('-ip', '--insertion_point', type=str, default='center', help='center or left or partly_left')
parser.add_argument('-k', '--kappa',type=float)
parser.add_argument('-p', '--rho',type=float)
parser.add_argument('-r', '--replicate', type=int, default=0)



# Read in arguments
args = vars(parser.parse_args())
filename = args['filename']
N_nodes = args['N_nodes']
edge_len = args['edge_len']
ellipse_ratio = args['ellipse_ratio']
insertion_point = args['insertion_point']
#N_sinks = args['N_sinks']
gamma = args['gamma']
kappa = args['kappa']
rho = args['rho']
replicate = args['replicate']
beta = 1.0 / (1 + gamma)
use_triangular_lattice = False

'''
# This input results in an average of 30 sinks/villous trees
N_sinks = 90
if ellipse_ratio==1.0:
    N_sinks = 40
'''
# This input results in an average of 40 sinks/villous trees
N_sinks = 116
if ellipse_ratio==1.0:
    N_sinks = 59


# Create a directory to store the data for these input parameters
run_dir = '../data/' + filename + f'_el{ellipse_ratio}_ip{insertion_point}'
continuing = os.path.exists(run_dir) #is this a continuation of a previous run?

# If this is a new run
if not continuing:
    os.mkdir(run_dir)

    # Store input parameters in a file in the directory we create for this dataset
    #(note: this will also store kappa/rho/replicate, but those vary for different runs)
    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"

    with open(run_dir + "/params.txt", "w+") as f:
        f.write(params_str)


# Create a network with the specified number of nodes (Note: we will cut circular/elliptical subsections from this network)
if use_triangular_lattice:
    nodes = triangular_lattice_pts(N_nodes, edge_len)
    edges = get_edges(nodes)
else:
    lattice_dir = '../lattices/'
    nodes = np.loadtxt(lattice_dir + 'nodes_10k_v2.txt')
    edges = np.loadtxt(lattice_dir + 'edges_10k_v2.txt')
netw_ = network_from_edges_and_nodes(edges, nodes)

if ellipse_ratio==1.0:
    circle_cutoff = 0.35
else:
    circle_cutoff = 0.5
netw = make_ellipse_netw(netw_, circle_cutoff, circle_cutoff) #ensure final network structure has no edges
netw = make_ellipse_netw(netw, 1.0, ellipse_ratio)

inds = network_indices(netw)
source_ind = inds[insertion_point]


# Specify parameters of the ODE to loop over

if not continuing:
    os.mkdir(run_dir + '/Ks/')
    os.mkdir(run_dir + '/sink_nodes/')


k = kappa
p = rho
r = replicate

K_file = run_dir + f'/Ks/K_kappa{np.round(k, 5)}_rho{np.round(p, 5)}_replicate{r}.txt'
sink_file = run_dir + f'/sink_nodes/sink_nodes_kappa{np.round(k, 5)}_rho{np.round(p, 5)}_replicate{r}.txt'

if not os.path.exists(K_file):

    K0 = -np.log10(np.random.rand(netw.N_e))
    num_sinks = np.random.randint(N_sinks-5, N_sinks+5) #vary number of sinks within the usual biological range (30-40) (Note: get_sinks returns fewer than the specified #)

    sink_nodes = get_sinks(num_sinks, netw)
    print('num sinks: ', len(sink_nodes))
    currents = lambda K, netw: static_currents(K, netw, source_index=source_ind, sink_nodes=sink_nodes)
    K, converged = ss_solve(lambda K, t: adaptation_ode(K, t, netw, currents, k, beta, p), K0, Δt=1.0)
    if converged:
        print('Converged')

    np.savetxt(K_file, K)
    np.savetxt(sink_file, sink_nodes)






