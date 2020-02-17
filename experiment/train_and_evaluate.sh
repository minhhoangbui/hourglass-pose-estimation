#!/usr/bin/env bash
DATASET=se7en11
NUM_STACKS=2
NUM_BLOCKS=1
DATASET_FOLDER=/home/hoangbm/data/se7en11
ANNOTATION=/home/hoangbm/data/se7en11
CHECKPOINT_PATH=/mnt/ssd2/Users/hoangbm/checkpoint/pose-estimation
TRAIN_BATCH=72
TEST_BATCH=72
GPUS=0,1
EVALUATION=false
SCHEDULE='250 300'
SUBSET='4 5 6 7 8 9 10 11'
MOBILE=true
EPOCHS=350
RESUME=/mnt/ssd2/Users/hoangbm/checkpoint/pose-estimation/se7en11_s2_b1_mobile_all/checkpoint.pth.tar

KD_ALPHA=1.0
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
--blocks ${NUM_BLOCKS} --image-path ${DATASET_FOLDER} --anno-path ${ANNOTATION} --checkpoint ${CHECKPOINT_PATH} -j 8 \
--train-batch ${TRAIN_BATCH} --test-batch ${TEST_BATCH} --schedule ${SCHEDULE} --epochs ${EPOCHS}  --lr 2.5e-3 \
 ${EVALUATION} ${MOBILE} --resume ${RESUME}

#CUDA_VISIBLE_DEVICES=${GPUS}  python experiment/train_and_evaluate_kd.py --dataset ${DATASET}  --stacks ${NUM_STACKS} \
#--blocks ${NUM_BLOCKS} --image-path ${DATASET_FOLDER} --anno-path ${ANNOTATION} --checkpoint ${CHECKPOINT_PATH} -j 16 \
#--train-batch ${TRAIN_BATCH} --test-batch ${TEST_BATCH} --schedule ${SCHEDULE} --epochs ${EPOCHS}  --lr 2.5e-03 \
# ${EVALUATION} ${MOBILE} --teacher-checkpoint ${TEACHER_CHECKPOINT} --teacher-stacks ${TEACHER_STACKS} \
# --kdloss-alpha ${KD_ALPHA} --resume ${RESUME}

