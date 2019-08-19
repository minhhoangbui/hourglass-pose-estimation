#!/usr/bin/env bash
DATASET=lsp
NUM_STACKS=2
NUM_BLOCKS=1
DATASET_FOLDER=/mnt/hdd3tb/Download/lsp/
CHECKPOINT_PATH=/mnt/hdd3tb/Users/son/gccpm/checkpoints/hg_s2_b1/
BATCH_SIZE=32
GPUS=0
EVALUATION=false
MOBILE=true
EPOCHS=64

if [[ ${EVALUATION} == true ]]
then
    EVALUATION=-e
else
    EVALUATION=
fi

if [[ ${mobile} == true ]]
then
    EVALUATION=--mobile
else
    EVALUATION=
fi

CUDA_VISIBLE_DEVICES=${GPUS}  python experiment/train_and_evaluate.py --dataset ${DATASET}  --stacks ${NUM_STACKS} \
--blocks ${NUM_BLOCKS} --image-path ${DATASET_FOLDER} --anno-path ${ANNOTATION} --checkpoint ${CHECKPOINT_PATH} -j 4 \
--train-batch ${BATCH_SIZE} --test-batch ${BATCH_SIZE}  --schedule ${SCHEDULE} --epochs ${EPOCHS}  --lr 2.5e-3 \
 --resume ${RESUME} ${EVALUATION} ${MOBILE}