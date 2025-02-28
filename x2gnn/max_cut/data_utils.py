from torch_geometric.data import Data, Dataset
import pickle
import numpy as np
import torch
from numba import jit, njit
import os
import networkx as nx

class GraphDatasetLazy(Dataset):
    def __init__(self, data_list, k, constructor):
        super(GraphDatasetLazy, self).__init__()
        self.k = k
        self.data_list = data_list
        self.multi_graph_constructor = constructor
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        graph = self.data_list[idx]
        num_nodes = graph.shape[0]
        edge_index = np.array(np.where(graph == 1))
        edge_index_multi, cross_edgge_index, cross_edgge_index2 = self.multi_graph_constructor(graph, edge_index, self.k)
        multi_edge_index = torch.tensor(edge_index_multi, dtype=torch.long)
        cross_edge_index = torch.tensor(cross_edgge_index, dtype=torch.long)
        cross_edge_index2 = torch.tensor(cross_edgge_index2, dtype=torch.long)

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(0.5).view(1,1).repeat(num_nodes * self.k, 2)
        data = Data(x=x, edge_index=multi_edge_index, cross_edge_index=cross_edge_index, cross_edge_index2=cross_edge_index2, adj=graph)

        return data
    
class GraphDatasetLazyNetworkx(Dataset):
    def __init__(self, data_list, k, constructor):
        super(GraphDatasetLazyNetworkx, self).__init__()
        self.k = k
        self.data_list = data_list
        self.multi_graph_constructor = constructor
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        graph = self.data_list[idx]
        num_nodes = graph.number_of_nodes()
        edges = list(graph.edges)
        edge_index = np.array(torch.tensor(edges + [(v,u) for u,v in edges]).t().contiguous())
        edge_index_multi, cross_edgge_index, cross_edgge_index2 = self.multi_graph_constructor(graph, edge_index, self.k)
        multi_edge_index = torch.tensor(edge_index_multi, dtype=torch.long)
        cross_edge_index = torch.tensor(cross_edgge_index, dtype=torch.long)
        cross_edge_index2 = torch.tensor(cross_edgge_index2, dtype=torch.long)

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(0.5).view(1,1).repeat(num_nodes * self.k, 2)
        data = Data(x=x, edge_index=multi_edge_index, cross_edge_index=cross_edge_index, cross_edge_index2=cross_edge_index2, adj=graph)


        return data


def load_data(path):
    train_data = pickle.load(open(f'{path}/train.pkl', 'rb'))
    valid_data = pickle.load(open(f'{path}/test.pkl', 'rb'))
    
    for arr in train_data:
        np.fill_diagonal(arr, 0)

    for arr in valid_data:
        np.fill_diagonal(arr, 0)
            
    print('Train:', len(train_data), ' Test:', len(valid_data))
    return train_data, valid_data

def load_data_nx(path):
    train_data = pickle.load(open(f'{path}/train_graphs.pkl', 'rb'))
    valid_data = pickle.load(open(f'{path}/test_graphs.pkl', 'rb'))
    
            
    print('Train:', len(train_data), ' Test:', len(valid_data))
    return train_data, valid_data


def load_data_val(path):
    valid_data = pickle.load(open(f'{path}/test.pkl', 'rb'))
    

    for arr in valid_data:
        np.fill_diagonal(arr, 0)
            
    print('Test:', len(valid_data))
    return None, valid_data

@njit
def construct_multi_graph_complete(graph, edge_index, k):
    N = graph.shape[0]
    E = edge_index.shape[1]
    
    # Create multi_edge_index manually
    multi_edge_index = np.empty((2, E * k), dtype=edge_index.dtype)
    for i in range(k):
        multi_edge_index[:, i * E:(i + 1) * E] = edge_index + i * N
    
    # Create cross_edges manually
    cross_edges_list = []
    base = np.arange(N)
    for i in range(k):
        for j in range(i + 1, k):
            cross_edges_list.append(np.stack((base + i * N, base + j * N)))
            cross_edges_list.append(np.stack((base + j * N, base + i * N)))


    cross_edges_list2 = []
    base = np.arange(E)
    for i in range(k):
        for j in range(i + 1, k):
            cross_edges_list2.append(np.stack((base + i * E, base + j * E)))    
            cross_edges_list2.append(np.stack((base + j * E, base + i * E)))
    
    # Convert list of cross_edges to a single array
    total_cross_edges = len(cross_edges_list)
    cross_edges = np.empty((2, N * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list):
        cross_edges[:, idx * N:(idx + 1) * N] = cross_edge

    total_cross_edges = len(cross_edges_list2)
    cross_edges2 = np.empty((2, E * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list2):
        cross_edges2[:, idx * E:(idx + 1) * E] = cross_edge

    
    return multi_edge_index, cross_edges, cross_edges2

@njit
def construct_multi_graph_fast_pair(graph, edge_index, k):
    N = graph.shape[0]
    E = edge_index.shape[1]
    
    multi_edge_index = np.empty((2, E * k), dtype=edge_index.dtype)
    for i in range(k):
        multi_edge_index[:, i * E:(i + 1) * E] = edge_index + i * N
  
    cross_edges_list = []
    base = np.arange(N)
    for i in range(k//2):    
        cross_edges_list.append(np.stack((base + 2*i * N, base + (2*i+1) * N)))
        cross_edges_list.append(np.stack((base + (2*i+1) * N, base + 2*i * N)))
    
    cross_edges_list2 = []
    base = np.arange(E)
    for i in range(k//2):
        cross_edges_list2.append(np.stack((base + 2*i * E, base + (2*i+1) * E)))    
        cross_edges_list2.append(np.stack((base + (2*i+1) * E, base + (2*i) * E)))

    total_cross_edges = len(cross_edges_list)
    cross_edges = np.empty((2, N * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list):
        cross_edges[:, idx * N:(idx + 1) * N] = cross_edge

    total_cross_edges = len(cross_edges_list2)
    cross_edges2 = np.empty((2, E * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list2):
        cross_edges2[:, idx * E:(idx + 1) * E] = cross_edge
    
    
    return multi_edge_index, cross_edges, cross_edges2

@njit
def construct_multi_graph_fast_directed_pair(graph, edge_index, k):
    N = graph.shape[0]
    E = edge_index.shape[1]
    
    multi_edge_index = np.empty((2, E * k), dtype=edge_index.dtype)
    for i in range(k):
        multi_edge_index[:, i * E:(i + 1) * E] = edge_index + i * N
  
    cross_edges_list = []
    base = np.arange(N)
    for i in range(k//2):    
        cross_edges_list.append(np.stack((base + 2*i * N, base + (2*i+1) * N)))

    cross_edges_list2 = []
    base = np.arange(E)
    for i in range(k//2):
        cross_edges_list2.append(np.stack((base + 2*i * E, base + (2*i+1) * E)))    

    
    total_cross_edges = len(cross_edges_list)
    cross_edges = np.empty((2, N * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list):
        cross_edges[:, idx * N:(idx + 1) * N] = cross_edge

    total_cross_edges = len(cross_edges_list)
    cross_edges2 = np.empty((2, E * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list2):
        cross_edges2[:, idx * E:(idx + 1) * E] = cross_edge

    
    return multi_edge_index, cross_edges, cross_edges2


def construct_multi_graph_fast_pair_nx(graph, edge_index, k):
    N = graph.number_of_nodes()
    E = edge_index.shape[1]
    
    multi_edge_index = np.empty((2, E * k), dtype=edge_index.dtype)
    for i in range(k):
        multi_edge_index[:, i * E:(i + 1) * E] = edge_index + i * N
  
    cross_edges_list = []
    base = np.arange(N)
    for i in range(k//2):    
        cross_edges_list.append(np.stack((base + 2*i * N, base + (2*i+1) * N)))
        cross_edges_list.append(np.stack((base + (2*i+1) * N, base + 2*i * N)))
    
    cross_edges_list2 = []
    base = np.arange(E)
    for i in range(k//2):
        cross_edges_list2.append(np.stack((base + 2*i * E, base + (2*i+1) * E)))    
        cross_edges_list2.append(np.stack((base + (2*i+1) * E, base + (2*i) * E)))

    total_cross_edges = len(cross_edges_list)
    cross_edges = np.empty((2, N * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list):
        cross_edges[:, idx * N:(idx + 1) * N] = cross_edge

    total_cross_edges = len(cross_edges_list2)
    cross_edges2 = np.empty((2, E * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list2):
        cross_edges2[:, idx * E:(idx + 1) * E] = cross_edge
    
    
    return multi_edge_index, cross_edges, cross_edges2



@njit
def construct_multi_graph_fast_quadruples(graph, edge_index, k):
    N = graph.shape[0]
    E = edge_index.shape[1]
    
    multi_edge_index = np.empty((2, E * k), dtype=edge_index.dtype)
    for i in range(k):
        multi_edge_index[:, i * E:(i + 1) * E] = edge_index + i * N
  
    cross_edges_list = []
    base = np.arange(N)
    for i in range(k//4):
        for j in range(4):
            for j2 in range(4):
                if j != j2:
                    cross_edges_list.append(np.stack((base + (4*i+j) * N, base + (4*i+j2) * N)))

    cross_edges_list2 = []
    base = np.arange(E)
    for i in range(k//4):
        for j in range(4):
            for j2 in range(4):
                if j != j2:
                    cross_edges_list2.append(np.stack((base + (4*i+j) * E, base + (4*i+j2) * E)))
        


    total_cross_edges = len(cross_edges_list)
    cross_edges = np.empty((2, N * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list):
        cross_edges[:, idx * N:(idx + 1) * N] = cross_edge

    total_cross_edges = len(cross_edges_list2)
    cross_edges2 = np.empty((2, E * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list2):
        cross_edges2[:, idx * E:(idx + 1) * E] = cross_edge
    
    return multi_edge_index, cross_edges, cross_edges2



def load_txt_graph(file):
    lines = open(file, 'r').readlines()
    n = int(lines[0].split()[0])
    m = int(lines[0].split()[1])
    graph = np.zeros((n,n))
    for line in lines[1:]:
        u,v,w = line.split()
        assert(w == '1')
        u = int(u) - 1
        v = int(v) - 1
        graph[u][v] = 1
        graph[v][u] = 1

    return graph




def load_data_gset(main_path):
    graphs = []
    filenames = []
    for file in sorted(os.listdir(main_path)):
        filenames.append(file)
        graph = load_txt_graph(f'{main_path}/{file}')
        np.fill_diagonal(graph, 0)
        graphs.append(graph)

    return graphs, filenames


