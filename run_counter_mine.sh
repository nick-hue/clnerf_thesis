#!/bin/bash

# export DATA_DIR=/workspace/data/counter_part/counter_1_40/
# export DATA_DIR=/workspace/data/counter_sm_merged/
export DATA_DIR=/workspace/data/drz

task_curr=0
task_number=2 # task number has to be the same number as the total number of training parts available
scene_name=drz    
rep=5                   # default 10
epochs=10               # default 20
batch_size=4096         # default 8192
downsample=1.0          # default 1.0
dim_a=48                # default 48
dim_g=16                # default 16
scale=8.0               # default 8.0
lr=1e-2                 # default 1e-2
num_gpus=1

experiment_name=${scene_name}_s${scale}_lr${lr}_dima${dim_a}_dimg${dim_g}_r${rep}_e${epochs}_b${batch_size}_d${downsample}_gpu${num_gpus}
echo Experiment name   : $experiment_name
#--gpu2_render \

python train_ngpgv2_CLNerf.py \
        --root_dir $DATA_DIR \
        --dataset_name colmap_ngpa_CLNerf \
        --exp_name $experiment_name \
        --rep_size $rep \
        --num_epochs $epochs \
        --task_curr $task_curr \
        --task_number $task_number \
        --batch_size $batch_size \
        --num_gpus $num_gpus \
        --lr $lr \
        --eval_lpips \
        --dim_a $dim_a \
        --dim_g $dim_g \
        --scale $scale \
        --downsample ${downsample} \
        --vocab_size ${task_number} \
        --no_save_test # dont save test video and frames when done training 
        
echo Training done. 