#!/bin/bash
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=36g
#SBATCH --output=../experiment-results/cartography/pejorative/%x-%j.log
#SBATCH --job-name=cartography-pejorative-deberta-large
#SBATCH --account=linmacse0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dnaihao/dnaihao-scratch/anaconda3/envs/ann-embed/lib/
export CUDA_VISIBLE_DEVICES=0

dataset=pejorative
tasks=pejorative
method=add
include_pad_annotation=True
use_annotator_embed=True
use_annotation_embed=True

train_batch_size=64
num_train_epochs=3
learning_rate=1e-5
max_seq_length=256
warmup_steps=0
weight_decay=0.0

train_data_path=example-data/${dataset}-processed/annotation_split_train.json
annotator_id_path=example-data/${dataset}-processed/stats.json
annotation_label_path=example-data/${dataset}-processed/annotation_labels.json

SEEDS=(32 42 52 62 72 82 92 102 112 122)

mkdir -p ../experiment-results/cartography/pejorative

for i in {0..9}
do
    seed=${SEEDS[$i]}
    output_csv_seed=../experiment-results/cartography/pejorative/cartography_deberta_multichoice_pejorative_seed_${seed}.csv
    
    echo "Running cartography collection with seed $seed"
    
    python -m ablation_studies.cartography_deberta_multichoice \
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
        --output_csv ${output_csv_seed}
done
