import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
from numba import jit, njit
import time

from itertools import combinations


def compute_diversity_loss_v2(data, outputs, k):
    assert outputs.dim() == 2
    q = 2 * outputs[:,1] - 1
    A, B = data.cross_edge_index2[0], data.cross_edge_index2[1]
    C, D = data.edge_index[0], data.edge_index[1]
    cut_edges = ((1-q[C] * q[D])/2)
    diversity = torch.sum(cut_edges[A] * cut_edges[B])

    return diversity / data.num_graphs / k

def compute_diversity_loss(data, outputs, k):
    assert outputs.dim() == 2
    q = 2 * outputs[:,1] - 1
    A, B = data.cross_edge_index2[0], data.cross_edge_index2[1]
    C, D = data.edge_index[0], data.edge_index[1]
    cut_edges = ((1-q[C] * q[D])/2)
    diversity = torch.sum(cut_edges[A] * cut_edges[B])

    return diversity / data.num_graphs / k

def compute_cut_loss(data, outputs):
    assert outputs.dim() == 2
    q = 2 * outputs[:,1] - 1
    A, B = data.edge_index[0], data.edge_index[1]
    obj = torch.sum((1-q[A] * q[B])/2)

    return obj / data.num_graphs

def compute_obj_mc_separate(data, outputs, k):
    outputs2 = torch.argmax(outputs, dim=-1)#.detach().cpu().numpy()
    mean_objs, best_objs = extract_ind_sol_k(data, outputs2, k)
    return mean_objs, best_objs

@njit
def compute_cut_size_numba(solution, graph):
    #print('graph type', type(graph))
    #print(graph)
    cut_size = 0
    for i in range(graph.shape[0]):
        for j in range(i+1, graph.shape[0]):
            if graph[i,j] == 1:
                cut_size += solution[i] != solution[j]
    return cut_size

#@njit
def compute_obj_mc_separate_dynamic_threshold(data, outputs, k):
    #print('outputs:', outputs.shape)
    objs = []
    for i in range(k):
        o = outputs[i][:,1]
        #get unique values in o
        unique_o = np.unique(o)
        unique_o = unique_o[(unique_o > .01) & (unique_o < .99)]
        #print(unique_o)

        #instead of argmax, make everything above the threshold 1 and less than the threshold 0 
        #so if o[i] is less than threshold, set it to 0, otherwise set it to 1
        best_obj = -1
        for threshold in unique_o:
            o2 = o.copy()
            o2[o2 < threshold] = 0
            o2[o2 >= threshold] = 1

            obj = compute_cut_size_numba(o2, data.adj[0])
            if obj > best_obj:
                best_obj = obj

        objs.append(best_obj)
            
            
    objs = np.array(objs)
    return [np.mean(objs)], [np.max(objs)]

def extract_ind_sol_k(data, outputs, k):
    cidx = 0
    objs = torch.zeros((len(data),k), device=outputs.device)
    for idx in range(len(data)):
        s, e = cidx,cidx+data[idx].x.shape[0]
        
        start = s
        part = (e - s + 1) // k
        s2 = 0
        part2 = (data[idx].edge_index.size(1) // k)
        for j in range(k):
            q = 2 * outputs[start:start+part] - 1
            A, B = data[idx].edge_index[0][s2:s2+part2], data[idx].edge_index[1][s2:s2+part2]
            objs[idx,j] = torch.sum((1-q[A] * q[B])/2) / 2
            start += part
        cidx += data[idx].x.shape[0]
        
    return np.mean(objs.cpu().numpy(), axis=1), np.max(objs.cpu().numpy(), axis=1)

