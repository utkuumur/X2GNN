from torch_geometric.data import Data, Dataset
import pickle
import numpy as np
import torch
from numba import jit, njit
import networkx as nx
from tqdm import tqdm

    
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
        edge_index_multi, cross_edgge_index = self.multi_graph_constructor(graph, edge_index, self.k)
        multi_edge_index = torch.tensor(edge_index_multi, dtype=torch.long)
        cross_edge_index = torch.tensor(cross_edgge_index, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(0.5).view(1,1).repeat(num_nodes * self.k, 2)
        data = Data(x=x, edge_index=multi_edge_index, cross_edge_index=cross_edge_index, adj=graph)

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
        edge_index_multi, cross_edgge_index = self.multi_graph_constructor(graph, edge_index, self.k)
        multi_edge_index = torch.tensor(edge_index_multi, dtype=torch.long)
        cross_edge_index = torch.tensor(cross_edgge_index, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(0.5).view(1,1).repeat(num_nodes * self.k, 2)
        data = Data(x=x, edge_index=multi_edge_index, cross_edge_index=cross_edge_index, adj=graph)

        return data
    
def load_data(path):
    train_data_org = pickle.load(open(f'{path}/train.pkl', 'rb'))
    valid_data_org = pickle.load(open(f'{path}/test.pkl', 'rb'))
    train_data = []
    valid_data = []

    for arr in train_data_org:
        arr2 = 1 - arr
        np.fill_diagonal(arr2, 0)
        train_data.append(arr2)

    for arr in valid_data_org:
        arr2 = 1 - arr
        np.fill_diagonal(arr2, 0)
        valid_data.append(arr2)
            
    print('Train:', len(train_data), ' Test:', len(valid_data))
    return train_data, valid_data

def load_data_nx(path):
    train_data = pickle.load(open(f'{path}/train_graphs.pkl', 'rb'))
    train_data = [nx.complement(g) for g in tqdm(train_data)]
    valid_data = pickle.load(open(f'{path}/test_graphs.pkl', 'rb'))
    valid_data = [nx.complement(g) for g in tqdm(valid_data)]
            
    print('Train:', len(train_data), ' Test:', len(valid_data))
    return train_data, valid_data

def load_data_val(path):
    valid_data = pickle.load(open(f'{path}/test.pkl', 'rb'))
    

    for arr in valid_data:
        np.fill_diagonal(arr, 0)
            
    print('Test:', len(valid_data))
    return None, valid_data

def construct_multi_graph_fast_pair_nx(graph, edge_index, k):
    N = graph.number_of_nodes()
    E = edge_index.shape[1]
    
    multi_edge_index = np.empty((2, E * k), dtype=np.int32)
    for i in range(k):
        multi_edge_index[:, i * E:(i + 1) * E] = edge_index + i * N
  
    cross_edges_list = []
    base = np.arange(N)
    for i in range(k//2):    
        cross_edges_list.append(np.stack((base + 2*i * N, base + (2*i+1) * N)))
        cross_edges_list.append(np.stack((base + 2*i+1 * N, base + 2*i * N)))
    
    total_cross_edges = len(cross_edges_list)
    cross_edges = np.empty((2, N * total_cross_edges), dtype=np.int32)
    for idx, cross_edge in enumerate(cross_edges_list):
        cross_edges[:, idx * N:(idx + 1) * N] = cross_edge
    
    return multi_edge_index, cross_edges


@njit
def construct_multi_graph_complete(graph, edge_index, k):
    N = graph.shape[0]
    E = edge_index.shape[1]
    
    multi_edge_index = np.empty((2, E * k), dtype=edge_index.dtype)
    for i in range(k):
        multi_edge_index[:, i * E:(i + 1) * E] = edge_index + i * N
  
    cross_edges_list = []
    base = np.arange(N)
    for i in range(k):
        for j in range(i + 1, k):
            cross_edges_list.append(np.stack((base + i * N, base + j * N)))
            cross_edges_list.append(np.stack((base + j * N, base + i * N)))
    
    total_cross_edges = len(cross_edges_list)
    cross_edges = np.empty((2, N * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list):
        cross_edges[:, idx * N:(idx + 1) * N] = cross_edge
    
    return multi_edge_index, cross_edges

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
    
    total_cross_edges = len(cross_edges_list)
    cross_edges = np.empty((2, N * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list):
        cross_edges[:, idx * N:(idx + 1) * N] = cross_edge
    
    return multi_edge_index, cross_edges

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
        cross_edges_list.append(np.stack((base + 2*i+1 * N, base + 2*i * N)))
    
    total_cross_edges = len(cross_edges_list)
    cross_edges = np.empty((2, N * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list):
        cross_edges[:, idx * N:(idx + 1) * N] = cross_edge
    
    return multi_edge_index, cross_edges



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


    total_cross_edges = len(cross_edges_list)
    cross_edges = np.empty((2, N * total_cross_edges), dtype=edge_index.dtype)
    for idx, cross_edge in enumerate(cross_edges_list):
        cross_edges[:, idx * N:(idx + 1) * N] = cross_edge
    
    return multi_edge_index, cross_edges