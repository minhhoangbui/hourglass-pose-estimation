#!/usr/bin/env bash
DATASET=merl3k
NUM_STACKS=8
NUM_BLOCKS=1
DATASET_FOLDER=/mnt/hdd3tb/Datasets/MERL3000/images
ANNOTATION=/mnt/hdd3tb/Datasets/MERL3000/annotation/settings/1
CHECKPOINT_PATH=/mnt/hdd3tb/Users/hoang/checkpoint/pose-estimation/
TRAIN_BATCH=32
TEST_BATCH=32
GPUS=1,0
EVALUATION=false
SCHEDULE='40 60'
MOBILE=false
EPOCHS=70
RESUME=/mnt/hdd3tb/Users/hoang/checkpoint/pose-estimation/merl3k_s8_b1_non-mobile_all/checkpoint.pth.tar

TEACHER_CHECKPOINT=/mnt/hdd10tb/Users/hoang/checkpoint/pose-estimation/mscoco_v2_s8_b1_non-mobile_all/model_best.pth.tar
TEACHER_STACKS=8

if [[ ${EVALUATION} == true ]]
then
    EVALUATION=-e
else
    EVALUATION=
fi
if [[ ${MOBILE} == true ]]
then
    MOBILE=--mobile
else
    MOBILE=
fi

CUDA_VISIBLE_DEVICES=${GPUS}  python experiment/train_and_evaluate.py --dataset ${DATASET}  --stacks ${NUM_STACKS} \
--blocks ${NUM_BLOCKS} --image-path ${DATASET_FOLDER} --anno-path ${ANNOTATION} --checkpoint ${CHECKPOINT_PATH} -j 16 \
--train-batch ${TRAIN_BATCH} --test-batch ${TEST_BATCH} --schedule ${SCHEDULE} --epochs ${EPOCHS}  --lr 2.5e-3 \
 ${EVALUATION} ${MOBILE} --resume ${RESUME}

#CUDA_VISIBLE_DEVICES=${GPUS}  python experiment/train_and_evaluate_kd.py --dataset ${DATASET}  --stacks ${NUM_STACKS} \
#--blocks ${NUM_BLOCKS} --image-path ${DATASET_FOLDER} --anno-path ${ANNOTATION} --checkpoint ${CHECKPOINT_PATH} -j 4 \
#--train-batch ${TRAIN_BATCH} --test-batch ${TEST_BATCH} --schedule ${SCHEDULE} --epochs ${EPOCHS}  --lr 2.5e-4 \
# ${EVALUATION} ${MOBILE} --resume ${RESUME} --teacher-checkpoint ${TEACHER_CHECKPOINT} --teacher-stacks ${TEACHER_STACKS}
