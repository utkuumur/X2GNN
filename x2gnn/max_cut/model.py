import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GINConv

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        
class GINCrossGATRes(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h, n_layer, layer_recurrence):
        super(GINCrossGATRes, self).__init__()
        self.wte = nn.Embedding(2, dim_h)
        self.k = 32
        self.layer_recurrence = layer_recurrence
        self.conv = nn.ModuleList([GINConv(
            nn.Sequential(nn.Linear(dim_h, dim_h),
                       LayerNorm(dim_h, bias=False), nn.ReLU(), 
                       nn.Linear(dim_h, dim_h),
                       LayerNorm(dim_h, bias=False), nn.ReLU()), train_eps=True) for _ in range(n_layer)])

        self.conv2 = nn.ModuleList([GATv2Conv(dim_h, dim_h, heads=1) for _ in range(n_layer)])
        self.batch_norms = nn.ModuleList([LayerNorm(dim_h, bias=False) for _ in range(n_layer)])

        self.lin1 = nn.Linear(dim_h*n_layer, dim_h)
        self.lin2 = nn.Linear(dim_h, 2)


    def forward(self, data, r):
        x, edge_index, cross_edge_index = data.x, data.edge_index, data.cross_edge_index
        y = x

        k = int (x.size(0) * 0.05)
        perm = torch.randperm(x.size(0),device=x.device)
        idx = perm[:k]
        y[idx, :] = torch.tensor([1.0, 0], device=x.device)

        h = torch.matmul(y, self.wte.weight)
        outputs = []
        
        for j in range(r):
            for i in range(self.layer_recurrence):
                hidden = []
                for idx, conv in enumerate(self.conv):
                    h = h+conv(h, edge_index)
                    h = h+self.conv2[idx](self.batch_norms[idx](h), cross_edge_index)
                    hidden.append(h)

                h = torch.cat(hidden, dim=1)
                h = self.lin1(h).relu()

                logits = self.lin2(h)
                y = F.softmax(logits, dim=1)
                outputs.append(y)



            y_drop = self.drop_and_normalize(y, data.prob)
            h = torch.matmul(y_drop, self.wte.weight)
            
            
        return outputs

    
    def drop_and_normalize(self, predictions, prob):
        T, N = predictions.shape
        all_zero = torch.tensor([1.0, 0.0], device=predictions.device).view(1,2).repeat(T,1)
        mask = torch.rand(T, device=predictions.device) < prob
        mask = mask.view(T,1).repeat(1, 2)
        return torch.where(mask, all_zero, predictions)

class GINCrossGATResEval(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h, n_layer, layer_recurrence):
        super(GINCrossGATResEval, self).__init__()
        self.wte = nn.Embedding(2, dim_h)
        self.layer_recurrence = layer_recurrence
        self.conv = nn.ModuleList([GINConv(
            nn.Sequential(nn.Linear(dim_h, dim_h),
                       LayerNorm(dim_h, bias=False), nn.ReLU(), 
                       nn.Linear(dim_h, dim_h),
                       LayerNorm(dim_h, bias=False), nn.ReLU()), train_eps=True) for _ in range(n_layer)])

        self.conv2 = nn.ModuleList([GATv2Conv(dim_h, dim_h, heads=1) for _ in range(n_layer)])
        self.batch_norms = nn.ModuleList([LayerNorm(dim_h, bias=False) for _ in range(n_layer)])
        self.lin1 = nn.Linear(dim_h*n_layer, dim_h)
        self.lin2 = nn.Linear(dim_h, 2)


    def forward(self, data, r):
        x, edge_index, cross_edge_index = data.x, data.edge_index, data.cross_edge_index
        y = x

        #sample 0.05 randomly
        k = int (x.size(0) * 0.05)
        perm = torch.randperm(x.size(0),device=x.device)
        idx = perm[:k]
        y[idx, :] = torch.tensor([1.0, 0], device=x.device)

        h = torch.matmul(y, self.wte.weight)
        outputs = []
        
        for j in range(r):
            for i in range(self.layer_recurrence):
                hidden = []
                for idx, conv in enumerate(self.conv):
                    h = h+conv(h, edge_index)
                    h = h+self.conv2[idx](self.batch_norms[idx](h), cross_edge_index)
                    hidden.append(h)

                h = torch.cat(hidden, dim=1)
                h = self.lin1(h).relu()

            logits = self.lin2(h)
            y = F.softmax(logits, dim=1)
            outputs.append(y)
          
            y_drop = self.drop_and_normalize(y, data.prob)
            h = torch.matmul(y_drop, self.wte.weight)
            
            
        return outputs
    
    def drop_and_normalize(self, predictions, prob):
        T, N = predictions.shape
        all_zero = torch.tensor([1.0, 0.0], device=predictions.device).view(1,2).repeat(T,1)
        mask = torch.rand(T, device=predictions.device) < prob
        mask = mask.view(T,1).repeat(1, 2)
        return torch.where(mask, all_zero, predictions)




#############################################################################################################
        
class GINRes(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h, n_layer, layer_recurrence):
        super(GINRes, self).__init__()
        self.wte = nn.Embedding(2, dim_h)
        self.layer_recurrence = layer_recurrence
        self.conv = nn.ModuleList([GINConv(
            nn.Sequential(nn.Linear(dim_h, dim_h),
                       LayerNorm(dim_h, bias=False), nn.ReLU(), 
                       nn.Linear(dim_h, dim_h),
                       LayerNorm(dim_h, bias=False), nn.ReLU()), train_eps=True) for _ in range(n_layer)])

        self.lin1 = nn.Linear(dim_h*n_layer, dim_h)
        self.lin2 = nn.Linear(dim_h, 2)


    def forward(self, data, r):
        x, edge_index, cross_edge_index = data.x, data.edge_index, data.cross_edge_index
        y = x

        k = int (x.size(0) * 0.05)
        perm = torch.randperm(x.size(0),device=x.device)
        idx = perm[:k]
        y[idx, :] = torch.tensor([1.0, 0], device=x.device)

        h = torch.matmul(y, self.wte.weight)
        outputs = []
        
        for j in range(r):
            for i in range(self.layer_recurrence):
                hidden = []
                for idx, conv in enumerate(self.conv):
                    h = h+conv(h, edge_index)
                    hidden.append(h)

                h = torch.cat(hidden, dim=1)
                h = self.lin1(h).relu()

                logits = self.lin2(h)
                y = F.softmax(logits, dim=1)
                outputs.append(y)



            y_drop = self.drop_and_normalize(y, data.prob)
            h = torch.matmul(y_drop, self.wte.weight)
            
            
        return outputs

    
    def drop_and_normalize(self, predictions, prob):
        T, N = predictions.shape
        all_zero = torch.tensor([1.0, 0.0], device=predictions.device).view(1,2).repeat(T,1)
        mask = torch.rand(T, device=predictions.device) < prob
        mask = mask.view(T,1).repeat(1, 2)
        return torch.where(mask, all_zero, predictions)



       
class GINResEval(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h, n_layer, layer_recurrence):
        super(GINResEval, self).__init__()
        self.wte = nn.Embedding(2, dim_h)
        self.layer_recurrence = layer_recurrence
        self.conv = nn.ModuleList([GINConv(
            nn.Sequential(nn.Linear(dim_h, dim_h),
                       LayerNorm(dim_h, bias=False), nn.ReLU(), 
                       nn.Linear(dim_h, dim_h),
                       LayerNorm(dim_h, bias=False), nn.ReLU()), train_eps=True) for _ in range(n_layer)])

        self.lin1 = nn.Linear(dim_h*n_layer, dim_h)
        self.lin2 = nn.Linear(dim_h, 2)


    def forward(self, data, r):
        x, edge_index, cross_edge_index = data.x, data.edge_index, data.cross_edge_index
        y = x

        k = int (x.size(0) * 0.05)
        perm = torch.randperm(x.size(0),device=x.device)
        idx = perm[:k]
        y[idx, :] = torch.tensor([1.0, 0], device=x.device)

        h = torch.matmul(y, self.wte.weight)
        outputs = []
        
        for j in range(r):
            for i in range(self.layer_recurrence):
                hidden = []
                for idx, conv in enumerate(self.conv):
                    h = h+conv(h, edge_index)
                    hidden.append(h)

                h = torch.cat(hidden, dim=1)
                h = self.lin1(h).relu()

                logits = self.lin2(h)
                y = F.softmax(logits, dim=1)
            
            
            outputs.append(y)
            y_drop = self.drop_and_normalize(y, data.prob)
            h = torch.matmul(y_drop, self.wte.weight)
            
            
        return outputs

    
    def drop_and_normalize(self, predictions, prob):
        T, N = predictions.shape
        all_zero = torch.tensor([1.0, 0.0], device=predictions.device).view(1,2).repeat(T,1)
        mask = torch.rand(T, device=predictions.device) < prob
        mask = mask.view(T,1).repeat(1, 2)
        return torch.where(mask, all_zero, predictions)
