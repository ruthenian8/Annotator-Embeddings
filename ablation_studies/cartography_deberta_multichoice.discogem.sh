#!/bin/bash
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=36g
#SBATCH --job-name=cartography-sentiment-deberta-v3-base
#SBATCH --account=linmacse0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dnaihao/dnaihao-scratch/anaconda3/envs/ann-embed/lib/
export CUDA_VISIBLE_DEVICES=0

model_name=deberta_multichoice
dataset=discogem1
tasks=discogem1
method=add
include_pad_annotation=True

train_batch_size=256
num_train_epochs=5
learning_rate=1e-5
max_seq_length=128
warmup_steps=0
weight_decay=0.0

train_data_path=../src/example-data/${dataset}-processed/annotation_split_train.json
annotator_id_path=../src/example-data/${dataset}-processed/annotator_ids.json
annotation_label_path=../src/example-data/${dataset}-processed/annotation_labels.json
output_csv=../experiment-results/cartography/${dataset}/cartography_deberta_multichoice_${dataset}.csv
wandb_name=${model_name}-${dataset}

SEEDS=(32 42 52 62 72 82 92 102 112 122)

mkdir -p ../experiment-results/cartography/${dataset}

for i in 0
do
    seed=${SEEDS[$i]}
    output_csv_seed=../experiment-results/cartography/${dataset}/cartography_deberta_multichoice_${dataset}_seed_${seed}.csv
    
    echo "Running cartography collection with seed $seed"
    
    python cartography_deberta_multichoice.py \
        --train_data_path ${train_data_path} \
        --annotator_id_path ${annotator_id_path} \
        --annotation_label_path ${annotation_label_path} \
        --tasks ${tasks} \
        --train_batch_size ${train_batch_size} \
        --num_train_epochs ${num_train_epochs} \
        --learning_rate ${learning_rate} \
        --max_seq_length ${max_seq_length} \
        --warmup_steps ${warmup_steps} \
        --weight_decay ${weight_decay} \
        --method ${method} \
        --include_pad_annotation \
        --seed ${seed} \
        --wandb_name ${wandb_name} \
        --output_csv ${output_csv_seed}
done
