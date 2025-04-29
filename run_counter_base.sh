#!/bin/bash

# this is for the smaller input so change the counter data to counter_sm

export DATA_DIR=/workspace/data/counter_sm # where counter dataset is saved

task_number=1
scene_name=counter
downsample=1.0
epochs=20
rep=10

experiment_name=${scene_name}_r${rep}_e${epochs}_t${task_number}

echo Now training experiment with name : $experiment_name


for ((i=0; i<$task_number; i++))
do
    python train_ngpgv2_CLNerf.py \
        --root_dir $DATA_DIR --dataset_name colmap_ngpa_CLNerf \
        --exp_name $experiment_name \
        --num_epochs $epochs --batch_size 8192 --lr 1e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number --dim_a 48 --scale 8.0 --downsample ${downsample} --vocab_size ${task_number}
done