#!/usr/bin/env bash
DATASET=mscoco_v2
NUM_STACKS=2
NUM_BLOCKS=1
DATASET_FOLDER=/home/hoang/datasets/COCO2017/images/
ANNOTATION=/home/hoang/datasets/COCO2017/annotations/
CHECKPOINT_PATH=/mnt/hdd10tb/Users/hoang/checkpoint/pose-estimation/
TRAIN_BATCH=48
TEST_BATCH=48
GPUS=2,3,1,0
EVALUATION=false
SCHEDULE='40 70'
MOBILE=true
EPOCHS=90
RESUME=/mnt/hdd10tb/Users/hoang/checkpoint/pose-estimation/mscoco_v2_s8_s2_mobile_all_1.0/checkpoint.pth.tar

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

#CUDA_VISIBLE_DEVICES=${GPUS}  python experiment/train_and_evaluate.py --dataset ${DATASET}  --stacks ${NUM_STACKS} \
#--blocks ${NUM_BLOCKS} --image-path ${DATASET_FOLDER} --anno-path ${ANNOTATION} --checkpoint ${CHECKPOINT_PATH} -j 8 \
#--train-batch ${TRAIN_BATCH} --test-batch ${TEST_BATCH} --schedule ${SCHEDULE} --epochs ${EPOCHS}  --lr 2.5e-3 \
# ${EVALUATION} ${MOBILE}

CUDA_VISIBLE_DEVICES=${GPUS}  python experiment/train_and_evaluate_kd.py --dataset ${DATASET}  --stacks ${NUM_STACKS} \
--blocks ${NUM_BLOCKS} --image-path ${DATASET_FOLDER} --anno-path ${ANNOTATION} --checkpoint ${CHECKPOINT_PATH} -j 16 \
--train-batch ${TRAIN_BATCH} --test-batch ${TEST_BATCH} --schedule ${SCHEDULE} --epochs ${EPOCHS}  --lr 2.5e-03 \
 ${EVALUATION} ${MOBILE} --teacher-checkpoint ${TEACHER_CHECKPOINT} --teacher-stacks ${TEACHER_STACKS} \
 --kdloss-alpha ${KD_ALPHA} --resume ${RESUME}

