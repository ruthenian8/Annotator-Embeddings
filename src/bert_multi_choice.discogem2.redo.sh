#!/bin/bash
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=36g
#SBATCH --output=/home/dnaihao/dnaihao-scratch/annotator-embedding/experiment-results/hs_brexit/roberta-base-naiive-concat/%x-%j.log
#SBATCH --job-name=hs_brexit
#SBATCH --account=mihalcea98

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dnaihao/dnaihao-scratch/anaconda3/envs/ann-embed/lib/
export CUDA_VISIBLE_DEVICES=0

##############################
#  Experiment‐wide settings  #
##############################

model_name=roberta-multichoice
dataset=discogem2
tasks=discogem2

broadcast_annotation_embedding=False
broadcast_annotator_embedding=False
include_pad_annotation=True
method=add

# We are only doing testing, so test_mode can be left as "normal"
# since the paradigm is entirely "test_from_scratch"
test_mode=normal

use_annotator_embed=True
use_annotation_embed=True

train_batch_size=72
eval_batch_size=72

# We won’t actually train—just load checkpoints:
output_ckpt_dir=ckpts/${model_name}.redo/text_finetuned/
num_train_epochs=0      # no training
wandb_name=${model_name}-${dataset}
log_dir=logs/${model_name}/
log_path=${log_dir}/baseline.log

model_name_or_path=roberta-base
annotator_id_path=example-data/${dataset}-processed/annotator_ids.json
annotation_label_path=example-data/${dataset}-processed/annotation_labels.json

# Where the data is:
train_data_path=example-data/${dataset}-processed/annotation_split_train.json
dev_data_path=example-data/${dataset}-processed/annotation_split_test.json
test_data_path=example-data/${dataset}-processed/annotation_split_test.json

# Create any directories
mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}

####################################################
#  Loop over checkpoints 4 through 21 and test     #
####################################################
for ckpt_number in {10..12}
do
    # Construct full path to the .ckpt file for this version
    ckpt_path="/home/dignatev/Annotator-Embeddings/src/transformer_models/lightning_logs/version_${ckpt_number}/checkpoints/epoch=2-step=813.ckpt"
    
    # Name the output files based on checkpoint version
    output_fn="version_${ckpt_number}_redo_test_results"
    pred_fn_path="../experiment-results/${dataset}/${model_name}/${output_fn}.jsonl"
    
    echo "==============================================="
    echo " Testing checkpoint version ${ckpt_number} "
    echo "  ->  ckpt:  ${ckpt_path}"
    echo "  ->  preds: ${pred_fn_path}"
    echo "==============================================="
    
    # If predictions already exist for this version, skip
    if [ -e "${pred_fn_path}" ]; then
        echo "File exists: ${pred_fn_path}  →  skipping."
        echo
        continue
    fi
    
    # Run the test_from_scratch paradigm on this checkpoint
    python -m src ${model_name} \
        --training_paradigm      test_from_scratch \
        --model_name_or_path     ${ckpt_path} \
        --train_data_path        ${train_data_path} \
        --dev_data_path          ${dev_data_path} \
        --test_data_path         ${test_data_path} \
        --train_batch_size       ${train_batch_size} \
        --eval_batch_size        ${eval_batch_size} \
        --add_output_tokens      True \
        --output_ckpt_dir        ${output_ckpt_dir} \
        --num_train_epochs       ${num_train_epochs} \
        --wandb_name             ${wandb_name} \
        --n_gpu                  1 \
        --learning_rate          1e-5 \
        --linear_scheduler       False \
        --wandb_offline          \
        --tasks                  ${tasks} \
        --use_annotator_embed    ${use_annotator_embed} \
        --use_annotation_embed   ${use_annotation_embed} \
        --broadcast_annotation_embedding ${broadcast_annotation_embedding} \
        --broadcast_annotator_embedding  ${broadcast_annotator_embedding} \
        --annotator_id_path      ${annotator_id_path} \
        --annotation_label_path  ${annotation_label_path} \
        --pred_fn_path           ${pred_fn_path} \
        --include_pad_annotation ${include_pad_annotation} \
        --method                 ${method} \
        --test_mode              ${test_mode} \
        --enable_checkpointing   False   \
        --use_naiive_concat \
        --check_val_every_n_epoch 1
    echo
done


