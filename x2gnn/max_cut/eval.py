import torch
from torch_geometric.loader import DataLoader

from utils import compute_obj_mc_separate
from data_utils import load_data, GraphDatasetLazy, construct_multi_graph_fast_directed_pair, construct_multi_graph_complete, construct_multi_graph_fast_pair, construct_multi_graph_fast_quadruples, load_data_gset, load_data_nx, GraphDatasetLazyNetworkx, construct_multi_graph_fast_pair_nx
from model import GINCrossGATResEval

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

def eval_iteration(model, valid_dataloader, device, R, p, k):
    """
    Evaluate model performance over multiple recurrence steps.
    
    Args:
        model: The neural network model
        valid_dataloader: DataLoader for validation/test data
        device: Device to use for computations (cpu or cuda)
        R: Number of recurrence steps
        p: Probability parameter for evaluation
        k: Number of graphs in the multilayer graph
        
    Returns:
        tuple: Lists of objective values and best objective values for each recurrence step
    """
    obj_vals = [[] for _ in range(R)]
    best_obj_vals = [[] for _ in range(R)]   

    with torch.no_grad():
        for data in tqdm(valid_dataloader):
            data.prob = 1-p
            data = data.to(device)
            layer_probs = model(data, R)

            for r in range(R):
                mean_objs, best_objs = compute_obj_mc_separate(data, layer_probs[r], k)
                obj_vals[r].extend(mean_objs)
                best_obj_vals[r].extend(best_objs)
            
    return obj_vals, best_obj_vals         
   
def main(args):
    """
    Main function to evaluate model performance across multiple recurrence steps.
    
    Args:
        args: Command line arguments
    """
    # Create model identifiers
    dataset = args.data_path.split('/')[-1]
    model_identifier = f'MultiGraph_{args.graph_constructor}_Train_{dataset}_L{args.num_layers}_R{args.num_recurrences}_H{args.hid_dim}_k{args.k}_LR{args.lr}_BS{args.batch_size}_E{args.ep}_drop{args.target}_div{args.div_coef}' 
    model_identifier2 = f'MultiGraph_{args.graph_constructor}_Eval_{dataset}_L{args.num_layers}_R{args.num_recurrences}_H{args.hid_dim}_k{args.k}_LR{args.lr}_BS{args.batch_size}_E{args.ep}_drop{args.target}_div{args.div_coef}' 

    # Initialize wandb if specified
    wandb = None
    if args.wandb:
        try:
            import wandb
            logger.info(f'Model Id: {model_identifier2}')
            wandb.init(project='x2gnn_max_cut')
            wandb.run.name = model_identifier2
        except ImportError:
            logger.warning("wandb not installed. Running without wandb logging.")
            args.wandb = False

    # Handle custom data path for larger dataset evaluation if needed
    eval_data_path = args.data_path
    
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

    # Load data and create test dataset based on data format
    logger.info(f"Loading data from {eval_data_path}")
    filenames = None
    if args.data_format == 'numpy':
        _, test_graphs = load_data(eval_data_path)
        eval_dataset = GraphDatasetLazy(test_graphs[:args.eval_data_size], args.k2, constructor)
    elif args.data_format == 'networkx':
        assert args.graph_constructor == 'undirected_pair', "Only undirected pair constructor is supported for networkx graphs"
        _, test_graphs = load_data_nx(eval_data_path)
        constructor = construct_multi_graph_fast_pair_nx
        eval_dataset = GraphDatasetLazyNetworkx(test_graphs[:args.eval_data_size], args.k2, constructor)
    elif args.data_format == 'txt' and 'GSET' in eval_data_path:
        graphs, filenames = load_data_gset(eval_data_path)
        eval_dataset = GraphDatasetLazy(graphs, args.k2, constructor)
        
        
    
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size // args.k2, 
                              shuffle=False, drop_last=False, pin_memory=True)

    # Initialize model architecture
    model = GINCrossGATResEval(args.hid_dim, args.num_layers, args.num_recurrences)
    
    # Setup device (CPU or GPU)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")

    # Load model weights
    model_path = args.model_path
    if not model_path:
        model_path = f'../../models/max_cut/{dataset}/{dataset}.pt'
    
    logger.info(f"Loading model from: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
        
    model = model.to(device)

    # Run evaluation
    logger.info(f"Evaluating model with R={args.R} recurrence steps")
    obj_vals, best_obj_vals = eval_iteration(model, eval_dataloader, device, args.R, args.target, args.k2)
    
    # Analyze results across different numbers of recurrence steps
    all_results = []
    for i in range(args.R):
        comb = np.asarray(best_obj_vals[i])
        all_results.append(comb)

    all_results = np.vstack(all_results)

    # Report results for different numbers of recurrence steps
    logger.info("Results for different numbers of recurrence steps:")
    for i in [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023]:
        if i < args.R:
            mean_best = np.mean(np.max(all_results[:i+1,:], axis=0))
            logger.info(f'R={i+1} Best: {mean_best:.3f}')
            
            # Log to wandb if enabled
            if args.wandb:
                wandb.log({f'Best_R_{i+1}': mean_best})

    # Compute and report overall best score
    best = np.max(all_results, axis=0)
    score = np.mean(best)
    logger.info(f'Overall Best Mean Objective: {score:.3f}')
    
    if args.wandb: 
        wandb.log({'Validation Mean Objective Refined': score})

    # Print individual instance results if requested
    if args.print_ind_results:
        logger.info("Individual instance results:")
        all_results = all_results.T
        for i in range(all_results.shape[0]):
            if filenames is not None:
                logger.info(f'Instance {filenames[i]}: Size={all_results[i].max():.1f}')
            else:
                logger.info(f'Instance {i}: Size={all_results[i].max():.1f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Max Cut model performance with multiple recurrence steps')
    # Training parameters (for model identification)
    parser.add_argument('--ep', type=int, default=600, help='Number of epochs used in training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate used in training')
    
    # Model configuration
    parser.add_argument('--target', type=float, default=0.8, help='Probability parameter for evaluation')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GIN and GAT Layers in the model')
    parser.add_argument('--k', type=int, default=1, help='Number of graphs in the multilayer graph (for model identifier)')
    parser.add_argument('--k2', type=int, default=1, help='Number of graphs in the multilayer graph (for evaluation)')
    parser.add_argument('--num_recurrences', type=int, default=1, help='Number of recurrence steps in model architecture')
    parser.add_argument('--hid_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--div_coef', type=float, default=0.5, help='Diversity loss coefficient')
    
    # Evaluation settings
    parser.add_argument('--R', type=int, default=32, help='Maximum number of recurrence steps for evaluation')
    parser.add_argument('--eval_data_size', type=int, default=500, help='Number of instances to evaluate')
    parser.add_argument('--large_eval', action='store_true', help='Use larger dataset for evaluation')
    parser.add_argument('--print_ind_results', action='store_true', help='Print results for individual instances')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for computation: "cpu", "cuda", or "auto"')
    
    # Data and model paths
    parser.add_argument('--data_path', type=str, required=True, help='Path for the graph data')
    parser.add_argument('--model_path', type=str, default='', help='Path to the model checkpoint')
    parser.add_argument('--graph_constructor', type=str, required=True, 
                        choices=['directed_pair', 'undirected_pair', 'complete', 'quad'],
                        help='Multilayer graph constructor type')
    parser.add_argument('--data_format', type=str, default='numpy', 
                        choices=['numpy', 'networkx', 'txt'],
                        help='Format of the input data')
    
    # Misc options
    parser.add_argument('--save', action='store_true', help='Save evaluation results')
    parser.add_argument('--wandb', action='store_true', help='Log results to wandb')
    parser.add_argument('--single_stage', action='store_true', help='Use single stage training model')

    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # Enable higher precision matrix multiplications if available
    torch.set_float32_matmul_precision('high')
    
    logger.info(f"Arguments: {args}")
    main(args)