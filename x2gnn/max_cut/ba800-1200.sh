#!/bin/bash
#SBATCH --partition=full 
#SBATCH --job-name=max_cut_large
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:a100:1

conda activate x2gnn

DATASET=ba800-1200



mkdir -p logs/$DATASET

# train the model on ba200-300 and select the best model based on a subset of the training data
python train.py        --num_layers 4 --num_recurrences 4 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --save 1 --k 8 --data_path ../../data/$DATASET --wandb --div_coef 0.75 --graph_constructor undirected_pair --model_path ../../models/max_cut/$DATASET   > logs/$DATASET/train.log 2>&1
python select_model.py --num_layers 4 --num_recurrences 4 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --save --k 8 --R 32 --data_path ../../data/$DATASET --div_coef 0.75 --graph_constructor undirected_pair --model_path ../../models/max_cut/$DATASET   > logs/$DATASET/select.log 2>&1

python eval.py         --num_layers 4 --num_recurrences 4 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --k 8 --k2 8 --div_coef 0.75 --data_path ../../data/$DATASET --R 256 --eval_data_size 500  --graph_constructor undirected_pair --model_path ../../models/max_cut/$DATASET/$DATASET.pt   > logs/$DATASET/eval_8x256.log 2>&1
python eval.py         --num_layers 4 --num_recurrences 4 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --k 8 --k2 32 --div_coef 0.75 --data_path ../../data/$DATASET --R 1024 --eval_data_size 500  --graph_constructor undirected_pair --model_path ../../models/max_cut/$DATASET/$DATASET.pt   > logs/$DATASET/eval_32x1024.log 2>&1
