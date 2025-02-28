import torch
from utils import compute_diversity_loss, compute_cut_loss, compute_obj_mc_separate
from data_utils import load_data, GraphDatasetLazy, construct_multi_graph_fast_directed_pair, construct_multi_graph_complete, construct_multi_graph_fast_pair, construct_multi_graph_fast_quadruples, load_data_nx, GraphDatasetLazyNetworkx, construct_multi_graph_fast_pair_nx

from model import GINCrossGATRes, GINRes
from torch_geometric.loader import DataLoader
import numpy as np
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_iteration(model, train_dataloader, optimizer, scheduler, device, R, p, DC, k):
    """
    Train the model for one iteration over the training dataset.
    
    Args:
        model: The neural network model
        train_dataloader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        device: Device to use for computations (cpu or cuda)
        R: Number of recurrence steps
        p: Probability parameter for training
        DC: Diversity coefficient for loss computation
        k: Number of graphs in the multilayer graph
        
    Returns:
        tuple: (mean objective value, mean refined objective value)
    """
    model.train()
    train_objectives = []
    train_objectives2 = []

    for data in train_dataloader:
        loss = 0
        data.prob = p
        data = data.to(device)
    
        optimizer.zero_grad()

        layer_probs = model(data, R)
        for lp in layer_probs:
            obj_loss = compute_cut_loss(data, lp)        
            div_loss = compute_diversity_loss(data, lp, k)
            # Combine loss components: maximize MIS objective, minimize penalty and diversity loss
            loss += -obj_loss  + DC * div_loss
              
        # Backpropagation and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()

        # Compute metrics for tracking progress
        objs, objs2 = compute_obj_mc_separate(data, layer_probs[-1], k)
        train_objectives.extend(objs)
        train_objectives2.extend(objs2)

    # Calculate average metrics
    train_objectives = np.mean(train_objectives)
    train_objectives2 = np.mean(train_objectives2)
    return train_objectives, train_objectives2
    
def eval_iteration(model, valid_dataloader, optimizer, scheduler, device, R, p, DC, k):
    """
    Evaluate the model for one iteration over the validation dataset.
    
    Args:
        model: The neural network model
        valid_dataloader: DataLoader for validation data
        optimizer: Optimizer (not used during evaluation)
        scheduler: Learning rate scheduler (not used during evaluation)
        device: Device to use for computations (cpu or cuda)
        R: Number of recurrence steps
        p: Probability parameter for evaluation
        DC: Diversity coefficient for loss computation
        k: Number of graphs in the multilayer graph
        
    Returns:
        tuple: Multiple evaluation metrics including objectives and losses
    """
    model.train()  # Note: Model is set to train mode for evaluation
    #valid_objectives = []
    valid_objectives2 = []
    valid_objectives3 = []
    #valid_pens = []

    val_loss = 0.0
    val_pen_loss = 0.0
    val_ent_loss = 0.0
    val_mis_loss = 0.0
    val_div_loss = 0.0

    with torch.no_grad():
        for data in valid_dataloader:
            data.prob = p
            data = data.to(device)
            layer_probs = model(data, R)[-1]  # Take the last layer's probabilities
            
            # Compute various loss components
            obj_loss = compute_cut_loss(data, layer_probs)
            div_loss = compute_diversity_loss(data, layer_probs, k)
        
            # Accumulate losses
            val_loss += (-obj_loss)
            val_mis_loss -= obj_loss
            val_div_loss += div_loss
            
            # Compute objective metrics
            objs2, obsj2best = compute_obj_mc_separate(data, layer_probs, k)
            
            valid_objectives2.extend(objs2)
            valid_objectives3.extend(obsj2best)
            
    # Calculate average metrics
    z = (len(valid_dataloader) + 1)
    valid_objectives2, valid_objectives3 = np.mean(valid_objectives2), np.mean(valid_objectives3)

    #valid_objectives = torch.mean(torch.stack(valid_objectives)).item()
    #valid_pens = torch.mean(torch.stack(valid_pens)).item()
    
    return valid_objectives2, valid_objectives3, val_loss.item() / z, val_mis_loss.item() / z, 0, val_div_loss.item() / z 
  
def main(args):
    """
    Main function to run the training and evaluation process.
    
    Args:
        args: Command line arguments
    """
    # Create unique model identifier based on hyperparameters
    dataset = args.data_path.split('/')[-1]
    model_identifier = f'MultiGraph_{args.graph_constructor}_Train_{dataset}_L{args.num_layers}_R{args.num_recurrences}_H{args.hid_dim}_k{args.k}_LR{args.lr}_BS{args.batch_size}_E{args.ep}_drop{args.target}_div{args.div_coef}' 
   
    # Initialize wandb if specified
    wandb = None
    if args.wandb:
        try:
            import wandb
            logger.info(f'Model Id: {model_identifier}')
            wandb.init(project='x2gnn_max_cut')
            wandb.run.name = model_identifier
        except ImportError:
            logger.warning("wandb not installed. Running without wandb logging.")
            args.wandb = False

    # Select the appropriate graph constructor based on the argument
    constructor = None
    if args.graph_constructor == 'directed_pair':
        constructor = construct_multi_graph_fast_directed_pair
    elif args.graph_constructor == 'undirected_pair':
        constructor = construct_multi_graph_fast_pair
    elif args.graph_constructor == 'complete':
        constructor = construct_multi_graph_complete
    elif args.graph_constructor == 'quad':
        constructor = construct_multi_graph_fast_quadruples
    else:
        logger.error(f"Unknown graph constructor: {args.graph_constructor}")
        return

    # Load data and create datasets
    logger.info(f"Loading data from {args.data_path}")
    
    if args.data_format == 'numpy':
        train_graphs, val_graphs = load_data(args.data_path)
        train_dataset = GraphDatasetLazy(train_graphs, args.k, constructor)
        valid_dataset = GraphDatasetLazy(val_graphs, args.k, constructor)
    elif args.data_format == 'networkx':
        assert args.graph_constructor == 'undirected_pair', "Only undirected pair constructor is supported for networkx graphs"
        train_graphs, test_graphs = load_data_nx(args.data_path)
        constructor = construct_multi_graph_fast_pair_nx
        train_dataset = GraphDatasetLazyNetworkx(train_graphs, args.k, constructor)
        valid_dataset = GraphDatasetLazyNetworkx(test_graphs, args.k, constructor)
    
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size // args.k, 
                                 shuffle=True, drop_last=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size // args.k, 
                                 shuffle=True, drop_last=True, pin_memory=True)

    # Initialize model
    model = GINCrossGATRes(args.hid_dim, args.num_layers, args.num_recurrences)
    
    # Setup device (CPU or GPU)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    org_model = model.to(device)
    
    # Optionally compile the model for performance
    if args.compile:
        if hasattr(torch, 'compile'):
            logger.info("Using torch.compile for optimized performance")
            model = torch.compile(org_model, dynamic=True)
        else:
            logger.warning("torch.compile not available, using original model")
            model = org_model
    else:
        model = org_model

    # Setup optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = len(train_dataloader) * args.ep 
    warmup_steps = int(0.05 * total_steps)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                                        pct_start=warmup_steps / total_steps, anneal_strategy='linear')

    # Start wandb monitoring of model if enabled
    if args.wandb: 
        wandb.watch(model, log_freq=5)

    # First training phase
    logger.info("Starting first training phase")
    for epoch in tqdm(range(args.ep)):
        train_obj, train_obj2 = train_iteration(model, train_dataloader, optimizer, scheduler, device, 1, 0, args.div_coef)
        #valid_obj, valid_obj2, loss, loss_mis, valid_pen, loss_div = eval_iteration(model, valid_dataloader, optimizer, scheduler, device, 1, 0, args.div_coef)
        
        # no loss_end 
        valid_obj2, valid_obj3, loss, loss_mis, valid_pen, loss_div = eval_iteration(
            model, valid_dataloader, optimizer, scheduler, device, 1, 0, args.div_coef
        )    
        loss, loss_mis  = loss / args.k, loss_mis / args.k    
        
        
        
        # Log training progress
        logger.info(
            f'Epoch: {epoch} TrainMeanObjB={train_obj2:.1f} EvalMeanObjB={valid_obj3:.3f} '
            f'TrainMeanObjM={train_obj:.1f} EvalMeanObjM={valid_obj2:.3f} '
            f'EvalMeanPen={valid_pen:.1f} Loss={loss:.4f} MISLoss={loss_mis:.4f} '
            f'DivLoss={loss_div:.4f}'
        )
            
        # Log to wandb if enabled
        if args.wandb: 
            wandb.log({
                'Validation Mean Objective': valid_obj2, 
                'Validation Best Objective Refined': valid_obj3, 
                'Training Best Objective Refined': train_obj2, 
                'MIS Loss': loss_mis
            })
            
        # Save model if specified
        if args.save:
            sp = f'../../models/maxcut/{dataset}'   
            if args.model_path != '':
                sp = args.model_path
            os.makedirs(sp, exist_ok=True)
            torch.save(org_model.state_dict(), f'{sp}/weights_{epoch}.pt')

    # Setup for second training phase
    total_steps = len(train_dataloader) * args.ep 
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr / 4, total_steps=total_steps,
                                        pct_start=warmup_steps / total_steps, anneal_strategy='linear')

    # Calculate rate for probability decrease
    rate = 2 * (1 - args.target) / (args.ep)
    R = 2
    
    # Second training phase
    logger.info("Starting second training phase")
    for epoch in tqdm(range(args.ep)):
        # Adjust probability parameter
        p = max(args.target, 1-(epoch + 1) * rate)
        p = 1-p
    
        train_obj, train_obj2 = train_iteration(model, train_dataloader, optimizer, scheduler, device, R, p, args.div_coef)
        valid_obj2, valid_obj3, loss, loss_mis, valid_pen, loss_div = eval_iteration(
            model, valid_dataloader, optimizer, scheduler, device, 1, 0, args.div_coef
        )    
        loss, loss_mis  = loss / args.k, loss_mis / args.k    
        
        
         
        
        # Log training progress
        logger.info(
            f'Epoch: {epoch} TrainMeanObjB={train_obj2:.1f} EvalMeanObjB={valid_obj3:.3f} '
            f'TrainMeanObjM={train_obj:.1f} EvalMeanObjM={valid_obj2:.3f} '
            f'EvalMeanPen={valid_pen:.1f} Loss={loss:.4f} MISLoss={loss_mis:.4f} '
            f'DivLoss={loss_div:.4f}'
        )
            
        # Log to wandb if enabled
        if args.wandb: 
            wandb.log({
                'Validation Mean Objective': valid_obj2, 
                'Validation Best Objective Refined': valid_obj3, 
                'Training Best Objective Refined': train_obj2, 
                'MIS Loss': loss_mis
            })
            
            
        # Save model if specified
        if args.save:
            sp = f'../../models/maxcut/{dataset}'   
            if args.model_path != '':
                sp = args.model_path
            os.makedirs(sp, exist_ok=True)
            torch.save(org_model.state_dict(), f'{sp}/weights_second_{epoch}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Maximum Clique (MC) solver using neural networks')
    # Training
    parser.add_argument('--ep', type=int, default=600, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    # Model and loss
    parser.add_argument('--target', type=float, default=0.8, help='The probability of removing a node from solution between iterations')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GIN and GAT Layers in the model')
    parser.add_argument('--k', type=int, default=1, help='number of graphs in the multilayer graph')
    parser.add_argument('--num_recurrences', type=int, default=1, help='Number of recurrence steps in model.')
    parser.add_argument('--hid_dim', type=int, default=64, help='Hidden dimension size.')
    parser.add_argument('--save', type=int, default=0, help='Save model.')
    parser.add_argument('--wandb', default=False, action='store_true', help='save all logs on wandb')
    parser.add_argument('--compile', default=False, action='store_true', help='compile or not')
    parser.add_argument('--div_coef', type=float, default=0.75, help='Coefficient for the diversity loss term in the loss function')
    parser.add_argument('--data_path', type=str, default='', help='Path for the numpy adj matrix matrices or list of networkx graphs')
    parser.add_argument('--graph_constructor', type=str, default='', help='Multilayer graph constructor: [directed_pair, undirected_pair, complete]')
    parser.add_argument('--model_path', type=str, default='', help='Path for saving model weights')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for computation: "cpu", "cuda", or "auto" for automatic detection')
    parser.add_argument('--data_format', type=str, default='networkx', help='Format of the data: numpy or networkx')



    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # Enable higher precision matrix multiplications if available
    torch.set_float32_matmul_precision('high')
    
    logger.info(f"Arguments: {args}")
    main(args)