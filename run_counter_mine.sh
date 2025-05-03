#!/bin/bash

export DATA_DIR=/workspace/data/counter_sm_merged/ # where counter dataset is saved

task_curr=2
task_number=3
scene_name=counter
rep=10
epochs=10
downsample=0.5

experiment_name=${scene_name}_r${rep}_e${epochs}_t0_other
echo Experiment name   : $experiment_name

python train_ngpgv2_CLNerf.py \
        --root_dir $DATA_DIR \
        --dataset_name colmap_ngpa_CLNerf \
        --exp_name $experiment_name \
        --rep_size $rep \
        --num_epochs $epochs \
        --task_curr $task_curr \
        --task_number $task_number \
        --batch_size 8192 \
        --lr 1e-2 \
        --eval_lpips \
        --dim_a 48 \
        --scale 8.0 \
        --downsample ${downsample} \
        --vocab_size ${task_number} \
        --no_save_test