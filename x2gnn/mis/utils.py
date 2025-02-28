import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from numba import jit, njit


def compute_diversity_loss(data, outputs, k):
    q = outputs[:,1]
    A, B = data.cross_edge_index[0], data.cross_edge_index[1]
    norm = A.shape[0] / data.x.shape[0]
    diversity = torch.sum(q[A] * q[B])

    return diversity / data.num_graphs / norm / k

def compute_loss_entropy(data, outputs):
    entropy = 0
    q = outputs[:,1]
    q = torch.clamp(q, min=1e-6, max=1-1e-6)
    entropy += torch.sum((q * torch.log(q) + (1-q) * torch.log(1-q)))

    return entropy / data.num_graphs

def compute_mis_loss(data, outputs):
    q = outputs[:,1]
    A, B = data.edge_index[0], data.edge_index[1]
    obj = torch.sum(q)
    penalty = torch.sum(q[A] * q[B])

    return obj / data.num_graphs, penalty / 2 / data.num_graphs


def compute_mis_obj(data, outputs):
    outputs2 = torch.argmax(outputs, dim=-1)
    A, B = data.edge_index[0], data.edge_index[1]
    obj = torch.sum(outputs2)
    pen = torch.sum(outputs2[A] * outputs2[B])
    
    return obj / data.num_graphs, pen / 2 / data.num_graphs


def compute_obj_mis_refine(data, outputs, k):
    outputs2 = torch.argmax(outputs, dim=-1)
    refine_mis_vector(outputs2, outputs, data.edge_index[0], data.edge_index[1])
    mean_objs, best_objs = extract_ind_sol_k(data, outputs2, k)
    
    return mean_objs, best_objs

def extract_ind_sol_k(data, outputs, k):
    cidx = 0
    objs = torch.zeros((len(data),k))
    for idx in range(len(data)):
        s, e = cidx,cidx+data[idx].x.shape[0]
        start = s
        part = (e - s + 1) // k
        for j in range(k):
            objs[idx,j] = torch.sum(outputs[start:start+part])
            start += part
        cidx += data[idx].x.shape[0]
        
    return np.mean(objs.cpu().numpy(), axis=1), np.max(objs.cpu().numpy(), axis=1)






def refine_mis_vector(outputs, probs, A, B):
    mask = (outputs[A] + outputs[B] == 2)
    
    #handle the case with both endpoints
    prob_A = probs[A[mask], 1]
    prob_B = probs[B[mask], 1]
    change_A = prob_A < prob_B
    change_B = ~change_A  # the negation of change_A
    outputs2 = torch.zeros_like(outputs, device=outputs.device)
    outputs2[A[mask][change_A]] = 1
    outputs2[B[mask][change_B]] = 1
    mask2 = (outputs2[A] + outputs2[B] == 2)
    prob_A = probs[A[mask2], 1]
    prob_B = probs[B[mask2], 1]
    change_A = prob_A > prob_B
    change_B = ~change_A  # the negation of change_A

    # Update outputs: Remove one endpoint from the solution
    outputs[A[mask2][change_A]] = 0
    outputs[B[mask2][change_B]] = 0

    mask = (outputs[A] + outputs[B] == 2)
    prob_A = probs[A[mask], 1]
    prob_B = probs[B[mask], 1]

    change_A = prob_A < prob_B
    change_B = ~change_A
    outputs[A[mask][change_A]] = 0
    outputs[B[mask][change_B]] = 0

