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
