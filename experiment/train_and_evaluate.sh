#!/usr/bin/env bash
DATASET=lsp
NUM_STACKS=2
NUM_BLOCKS=1
#DATASET_FOLDER=/mnt/hdd10tb/Datasets/lsp/
#CHECKPOINT_PATH=/mnt/hdd10tb/Users/son/gccpm/checkpoints/hg_s2_b1_final/
DATASET_FOLDER=/mnt/hdd3tb/Download/lsp/
CHECKPOINT_PATH=/mnt/hdd3tb/Users/son/gccpm/checkpoints/hg_s2_b1/
NUM_WORKERS=4
BATCH_SIZE=32
GPUS=0
USE_GLOBAL_CONTEXT=true
EVALUATION=false

if [[ ${USE_GLOBAL_CONTEXT} == true ]]
then
    USE_GLOBAL_CONTEXT=--use-global-context
else
    USE_GLOBAL_CONTEXT=
fi

if [[ ${EVALUATION} == true ]]
then
    EVALUATION=-2
else
    EVALUATION=
fi

CUDA_VISIBLE_DEVICES=${GPUS} python experiment/train_and_evaluate.py --dataset ${DATASET} --stacks ${NUM_STACKS} \
--blocks ${NUM_BLOCKS} --image-path ${DATASET_FOLDER} --anno-path ${DATASET_FOLDER}/LEEDS_annotations.json \
--checkpoint ${CHECKPOINT_PATH} -j ${NUM_WORKERS} --train-batch ${BATCH_SIZE} --test-batch ${BATCH_SIZE} \
${USE_GLOBAL_CONTEXT} ${EVALUATION}