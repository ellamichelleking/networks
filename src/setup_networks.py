import numpy as np
from scipy import sparse


'''
Object that describes a network
    - pos: xyz positions of nodes
    - edgelist: list of edges
    - E: incidence matrix
    - N_e: Number of edges
    - N_v: Number of nodes (vertices)
    - mean_len = mean_len: average length of edges (measured by distances between of positions of nodes)
    - lengths: lengths of edges (measured by distances between of positions of nodes)
'''
class Network:
    def __init__(self, pos, edgelist, E, N_e, N_v, mean_len, lengths):
        self.pos = pos #xyz positions of nodes
        self.edgelist = edgelist #list of edges
        self.E = E #incidence matrix
        self.N_e = N_e #Number of edges
        self.N_v = N_v #Number of nodes (vertices)
        self.mean_len = mean_len #average length of edges (measured by distances between of positions of nodes)
        self.lengths = lengths #lengths of edges (measured by distances between of positions of nodes)

        
        
'''
Reads network from text into a Network object
    - edges: text file containing edges in the network
    - nodes: text file containing xyz positions of nodes in the network
'''        
def network_from_txt(edges, nodes, zoom=1.0):
    edgelist_float = np.loadtxt(edges, dtype=float)# + 1
    edgelist = edgelist_float.astype(int)

    sort_index = np.argsort(edgelist[:,0])
    edgelist = edgelist[sort_index]

    pos = np.loadtxt(nodes)
    
    N_e = len(edgelist) 
    N_v = len(pos)
    
    # remove overly long edges
    lengths = np.linalg.norm(pos[edgelist[:, 0]] - pos[edgelist[:, 1]], axis=1)
    mean_len = np.mean(lengths)

    J = np.tile(np.arange(N_e), 2)
    I = np.concatenate([edgelist[:,0], edgelist[:,1]])
    V = np.concatenate([np.ones(N_e), -1*np.ones(N_e)])

    E = sparse.csr_matrix((V, (I, J)), shape=(N_v, N_e))

    return Network(pos, edgelist, E, N_e, N_v, mean_len, lengths)


'''
Determine node indices for different sources: source placed at the center and
source placed at the lefthand side
'''
def network_indices(netw):
    """Return the index of the leftmost and center node"""
    i_left = np.argmin(netw.pos[:, 0])
    com = np.mean(netw.pos, axis=0)
    i_center = np.argmin([np.linalg.norm(netw.pos[i] - com) for i in range(netw.N_v)])

    return {"center": i_center, "left": i_left}

'''
Reads network from arrays into a Network object
    - edges: array containing edges in the network (list of pairs of IDs)
    - nodes: array containing xy(z) positions of nodes in the network
'''    
def network_from_edges_and_nodes(edges, pos, zoom=1.0):
    edges = np.array(edges)
    pos = np.array(pos)
    edgelist = edges.astype(int)
    N_e = len(edgelist) 
    N_v = len(pos)
    
    # remove overly long edges
    lengths = np.linalg.norm(pos[edgelist[:, 0]] - pos[edgelist[:, 1]], axis=1)
    mean_len = np.mean(lengths)

    J = np.tile(np.arange(N_e), 2)
    I = np.concatenate([edgelist[:,0], edgelist[:,1]])
    V = np.concatenate([np.ones(N_e), -1*np.ones(N_e)])

    E = sparse.csr_matrix((V, (I, J)), shape=(N_v, N_e))

    return Network(pos, edgelist, E, N_e, N_v, mean_len, lengths)




'''
Get xy coordinates for a triangular lattice with N_pts nodes and
a lattice spacing of lattice_spacing. Returns array of shape (N_pts, 2). 

Note that N_pts is assumed to be a perfect square; if it is not,
then an array of shape (sqrt(N_pts)**2, 2) will be returned.
'''
def triangular_lattice_pts(N_pts, lattice_spacing=0.5):
    N_pts_per_side = int(np.sqrt(N_pts))
    z_spacing = lattice_spacing*np.sqrt(3)/2
    
    tri = np.zeros((N_pts_per_side**2, 2))
    L = N_pts_per_side * lattice_spacing #Side length
    for i in range(N_pts_per_side):
        if i % 2 == 0:
            
            tri[i*N_pts_per_side:(i+1)*N_pts_per_side, 0] = np.arange(0, L, lattice_spacing)
        else:
            tri[i*N_pts_per_side:(i+1)*N_pts_per_side, 0] = np.arange(lattice_spacing/2, L+lattice_spacing/2, lattice_spacing)

        tri[i*N_pts_per_side:(i+1)*N_pts_per_side, 1] = np.full(N_pts_per_side, i*z_spacing)
        
    return tri


'''
Create list of edges given a list of 2D node positions.
Assumes that all neighboring points are connected by edges.
'''
def get_edges(node_positions):
    # Compute matrix of distances between all node positions
    dist_fn = lambda x: np.linalg.norm(x[np.newaxis, :, :] - x[:, np.newaxis, :], axis=2)
    dists = dist_fn(node_positions)
    lattice_spacing = np.min(dists[np.nonzero(dists)])
    
    # If distances are nonzero and less than 1.5*lattice_spacing, add an edge
    edges = []
    for i in range(len(dists)):
        di = dists[i]
        edge_candidates = np.where(di < 1.3*lattice_spacing)[0]
        for e in edge_candidates:
            if e > i:
                edges += [[i, e]]
            
    return np.array(edges)


'''
Save the nodes and edges from a Network object (netw) to two 
files with the names filename_nodes.npy and filename_edges.npy.
'''
def save_network(netw, filename):
    nodes = netw.pos
    edges = list(netw.edgelist)
    np.save(filename + '_nodes', nodes, allow_pickle=False)
    np.save(filename + '_edges', edges, allow_pickle=False)


'''
Having saved a network via the save_network method above, this 
function loads the network into a Network object and returns that object.
'''
def load_network(filename):
    nodes = np.load(filename + "_nodes.npy")
    edges = np.load(filename + "_edges.npy")
    return network_from_edges_and_nodes(edges, nodes)
