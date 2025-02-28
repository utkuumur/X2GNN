#!/bin/bash
#SBATCH --partition=full 
#SBATCH --job-name=clique_small
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:a100:1

conda activate x2gnn

DATASET=rb200-300
DATASET2=rb800-1200

mkdir -p logs/$DATASET
mkdir -p logs/$DATASET2

# uses the model trained on rb200-300

# evaluate the model on rb200-300
python eval.py         --num_layers 4 --num_recurrences 2 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --k 8 --k2 64 --div_coef 0.75 --data_path ../../data/$DATASET --R 32 --eval_data_size 500  --graph_constructor undirected_pair --model_path ../../models/clique/pretrained/$DATASET.pt   > logs/$DATASET/eval_64x32.log 2>&1
# evaluate the model on rb800-1200
python eval.py         --num_layers 4 --num_recurrences 2 --hid_dim 64 --batch_size 64 --ep 50 --lr 1e-3 --target 0.5 --k 8 --k2 64 --div_coef 0.75 --data_path ../../data/$DATASET2 --R 32 --eval_data_size 500  --graph_constructor undirected_pair --model_path ../../models/clique/pretrained/$DATASET.pt   > logs/$DATASET2/eval_64x32.log 2>&1


