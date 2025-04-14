#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1 # to use only the second gpu
export DATA_DIR=/workspace/data/ # where breville dataset is saved

task_number=2
scene_name=breville
downsample=1.0
epochs=20

rep=$1
for ((i=0; i<$task_number; i++))
do
    python train_ngpgv2_CLNerf.py \
        --root_dir $DATA_DIR --dataset_name colmap_ngpa_CLNerf \
        --exp_name ${scene_name}_r${rep}_e${epochs} \
        --num_epochs $epochs --batch_size 8192 --lr 1e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number --dim_a 48 --scale 8.0 --downsample ${downsample} --vocab_size ${task_number}
done