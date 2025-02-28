#!/bin/bash
#SBATCH --partition=full 
#SBATCH --job-name=mis_small
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:a100:1

conda activate x2gnn

DATASET=rb200-300
DATASET2=rb800-1200
DATASET3=er700-800



mkdir -p logs/$DATASET

# train the model on rb200-300 and select the best model based on a subset of the training data
python train.py        --num_layers 4 --num_recurrences 2 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --save 1 --k 8 --data_path ../../data/$DATASET --wandb --div_coef 0.75 --graph_constructor undirected_pair --model_path ../../models/mis/$DATASET   > logs/$DATASET/train.log 2>&1
python select_model.py --num_layers 4 --num_recurrences 2 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --save --k 8 --R 32 --data_path ../../data/$DATASET --div_coef 0.75 --graph_constructor undirected_pair --model_path ../../models/mis/$DATASET   > logs/$DATASET/select.log 2>&1

python eval.py         --num_layers 4 --num_recurrences 2 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --k 8 --k2 8 --div_coef 0.75 --data_path ../../data/$DATASET --R 256 --eval_data_size 500  --graph_constructor undirected_pair --model_path ../../models/mis/$DATASET/$DATASET.pt   > logs/$DATASET/eval_8x256.log 2>&1
python eval.py         --num_layers 4 --num_recurrences 2 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --k 8 --k2 32 --div_coef 0.75 --data_path ../../data/$DATASET --R 1024 --eval_data_size 500  --graph_constructor undirected_pair --model_path ../../models/mis/$DATASET/$DATASET.pt   > logs/$DATASET/eval_32x1024.log 2>&1

# generalization
# evaluate the model on rb800-1200
python eval.py         --num_layers 4 --num_recurrences 2 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --k 8 --k2 32 --div_coef 0.75 --data_path ../../data/$DATASET2 --R 256 --eval_data_size 500  --graph_constructor undirected_pair --model_path ../../models/mis/$DATASET/$DATASET.pt   > logs/$DATASET/eval_on_${DATASET2}_32x256.log 2>&1
# evaluate the model on er700-800
python eval.py         --num_layers 4 --num_recurrences 2 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --k 8 --k2 32 --div_coef 0.75 --data_path ../../data/$DATASET3 --R 256 --eval_data_size 500  --graph_constructor undirected_pair --model_path ../../models/mis/$DATASET/$DATASET.pt   > logs/$DATASET/eval_on_${DATASET3}_32x256.log 2>&1
