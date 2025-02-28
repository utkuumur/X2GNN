#!/bin/bash
#SBATCH --partition=full 
#SBATCH --job-name=mis_small
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:a100:1

conda activate x2gnn

# select the dataset from below
DATASET=rb200-300
#DATASET=rb800-1200
#DATASET=er700-800

mkdir -p logs/$DATASET

python eval.py         --num_layers 4 --num_recurrences 2 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --k 8 --k2 8 --div_coef 0.75 --data_path ../../data/$DATASET --R 256 --eval_data_size 500  --graph_constructor undirected_pair --model_path ../../models/mis/pretrained/$DATASET.pt   > logs/$DATASET/eval_8x256.log 2>&1
python eval.py         --num_layers 4 --num_recurrences 2 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --k 8 --k2 32 --div_coef 0.75 --data_path ../../data/$DATASET --R 1024 --eval_data_size 500  --graph_constructor undirected_pair --model_path ../../models/mis/pretrained/$DATASET.pt   > logs/$DATASET/eval_32x1024.log 2>&1
