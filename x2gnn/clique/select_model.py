import torch
from utils import compute_mis_obj, compute_mis_loss, compute_loss_entropy, compute_obj_mis_refine, compute_diversity_loss
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
import glob
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def eval_checkpoint(model, train_dataloader, device, R, k, p):
    """
    Evaluate a model checkpoint on the train dataset.
    
    Args:
        model: The neural network model
        train_dataloader: DataLoader for train data
        device: Device to use for computations (cpu or cuda)
        R: Number of recurrence steps
        k: Number of graphs in the multilayer graph
        
    Returns:
        tuple: Best mean objective value
    """
    model.eval()
    best_obj_vals = [[] for _ in range(R)]   

    with torch.no_grad():
        for data in train_dataloader:
            data.prob = 1-p
            data = data.to(device)
            
            # Forward pass through the model
            layer_probs = model(data, R)
            for r in range(R):
                mean_objs, best_objs = compute_obj_mis_refine(data, layer_probs[r], k)
                best_obj_vals[r].extend(best_objs)
                
    all = []
    for r in range(R):
        comb = np.asarray(best_obj_vals[r])
        all.append(comb)
    all = np.vstack(all)
    best_r = np.max(all, axis=0)
    score = np.mean(best_r)
                    
    return score

def main(args):
    """
    Main function to find the best checkpoint based on train set performance.
    
    Args:
        args: Command line arguments
    """
    # Create model identifier consistent with training script
    dataset = args.data_path.split('/')[-1]
    model_identifier = f'MultiGraph_{args.graph_constructor}_Train_{dataset}_L{args.num_layers}_R{args.num_recurrences}_H{args.hid_dim}_k{args.k}_LR{args.lr}_BS{args.batch_size}_E{args.ep}_drop{args.target}_div{args.div_coef}'
    
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

    # Load data and create train dataset
    logger.info(f"Loading data from {args.data_path}")
    if args.data_format == 'numpy':
        train_graphs, _ = load_data(args.data_path)
        train_dataset = GraphDatasetLazy(train_graphs[:args.eval_data_size], args.k, constructor)
    elif args.data_format == 'networkx':
        assert args.graph_constructor == 'undirected_pair', "Only undirected pair constructor is supported for networkx graphs"
        train_graphs, _ = load_data_nx(args.data_path)
        constructor = construct_multi_graph_fast_pair_nx
        train_dataset = GraphDatasetLazyNetworkx(train_graphs[:args.eval_data_size], args.k, constructor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size // args.k, 
                                 shuffle=False, drop_last=False, pin_memory=True)

    # Initialize model architecture (without loading weights yet)
    model = GINCrossGATRes(args.hid_dim, args.num_layers, args.num_recurrences)
    
    # Setup device (CPU or GPU)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Determine model weights directory
    weights_dir = f'../../models/clique/{dataset}/{model_identifier}' if args.model_path == '' else args.model_path
    logger.info(f"Looking for checkpoints in: {weights_dir}")
    
    # Find all checkpoint files from both training phases
    #phase1_checkpoints = sorted(glob.glob(f'{weights_dir}/weights_*.pt'))
    phase2_checkpoints = sorted(glob.glob(f'{weights_dir}/weights_second_*.pt'))
    #all_checkpoints = phase1_checkpoints + phase2_checkpoints
    all_checkpoints = phase2_checkpoints
    
    if not all_checkpoints:
        logger.error(f"No checkpoint files found in {weights_dir}")
        return
    
    logger.info(f"Found {len(all_checkpoints)} checkpoints to evaluate")
    
    # Track best checkpoint performance
    best_obj = float('-inf')
    best_checkpoint = None
    best_metrics = None
    
    # Evaluate each checkpoint
    for checkpoint_path in tqdm(all_checkpoints, desc="Evaluating checkpoints"):
        try:
            # Load checkpoint weights
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            
            # Get checkpoint epoch number from filename
            epoch = os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]
            phase = '1' if 'weights_second' not in checkpoint_path else '2'
            
            # Evaluate on test set
            bestMeanObj = eval_checkpoint(model, train_dataloader, device, args.R, args.k, args.target)

            
            # Log results
            logger.info(f"Phase {phase} Epoch {epoch}: TrainObjBestRefine={bestMeanObj:.3f}")
            
            # Track best checkpoint by refined objective
            if bestMeanObj > best_obj:
                best_obj = bestMeanObj
                best_checkpoint = checkpoint_path
                best_metrics = bestMeanObj
                
        except Exception as e:
            logger.error(f"Error evaluating checkpoint {checkpoint_path}: {str(e)}")
    
    # Report best checkpoint
    if best_checkpoint:
        logger.info(f"\nBest checkpoint: {best_checkpoint}")
        logger.info(
            f"Best metrics: TrainObjBestRefine={best_metrics:.3f}"
        )
        
        # Copy best checkpoint to final model path if requested
        if args.save:
            final_path = os.path.join(weights_dir, f"{dataset}.pt")
            try:
                # Load and save to ensure compatibility
                state_dict = torch.load(best_checkpoint, map_location=device)
                torch.save(state_dict, final_path)
                logger.info(f"Saved best model to {final_path}")
            except Exception as e:
                logger.error(f"Error saving best model: {str(e)}")
    else:
        logger.warning("No valid checkpoints found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select best checkpoint based on train performance')
    # Model configuration
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GIN and GAT Layers in the model')
    parser.add_argument('--k', type=int, default=1, help='Number of graphs in the multilayer graph')
    parser.add_argument('--num_recurrences', type=int, default=1, help='Number of recurrence steps in model')
    parser.add_argument('--hid_dim', type=int, default=64, help='Hidden dimension size')
    
    # Data and checkpoint paths
    parser.add_argument('--data_path', type=str, required=True, help='Path for the numpy adj matrix graphs')
    parser.add_argument('--model_path', type=str, default='', help='Path to directory containing checkpoints')
    parser.add_argument('--graph_constructor', type=str, required=True, 
                        choices=['directed_pair', 'undirected_pair', 'complete', 'quad'],
                        help='Multilayer graph constructor type')
    
    # Evaluation parameters
    parser.add_argument('--R', type=int, default=4, help='Maximum number of recurrence steps for evaluation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for computation: "cpu", "cuda", or "auto"')
    parser.add_argument('--save', action='store_true', help='Save best model to final path')
    
    # Parameters needed to reconstruct the model identifier
    parser.add_argument('--target', type=float, default=0.8, help='Probability from training')
    parser.add_argument('--div_coef', type=float, default=0.5, help='Diversity loss coefficient')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate from training')
    parser.add_argument('--ep', type=int, default=600, help='Number of training epochs')
    parser.add_argument('--data_format', type=str, default='networkx', help='Format of the data: numpy or networkx')
    parser.add_argument('--eval_data_size', type=int, default=320, help='Data size for eval')



    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # Log the configuration
    logger.info(f"Arguments: {args}")
    main(args)