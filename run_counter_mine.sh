#!/bin/bash

export DATA_DIR=/workspace/data/counter_sm_merged/ # where counter dataset is saved

task_curr=0
task_number=5 # task number has to be the same number as the total number of training parts available
scene_name=counter
rep=5
epochs=10
batch_size=2048
downsample=0.75

experiment_name=check_2048_${scene_name}_r${rep}_e${epochs}_b${batch_size}_d${downsample}
echo Experiment name   : $experiment_name

python train_ngpgv2_CLNerf.py \
        --root_dir $DATA_DIR \
        --dataset_name colmap_ngpa_CLNerf \
        --exp_name $experiment_name \
        --rep_size $rep \
        --num_epochs $epochs \
        --task_curr $task_curr \
        --task_number $task_number \
        --batch_size $batch_size \
        --lr 1e-2 \
        --eval_lpips \
        --dim_a 48 \
        --scale 8.0 \
        --downsample ${downsample} \
        --vocab_size ${task_number} \
        --no_save_test # dont save test video and frames when done training 
        
echo Training done. 