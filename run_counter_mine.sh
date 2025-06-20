#!/usr/bin/env bash

scene_name=counter_dark   # scene name
export DATA_DIR=/workspace/data/counter_final_final/$scene_name/

task_curr=1             # current task number, this is the task that is being trained
task_number=2           # task number has to be the same number as the total number of training parts available
rep=5                   # replay buffer size (default 10)
epochs=20               # epochs number during training (default 20)
batch_size=8192         # batch_size while training (default 8192)
downsample=1.0          # downsampling during rendering (default 1.0)
dim_a=48                # dimension of embeddings (default 48)
dim_g=16                # dimension of geometry embeddings (default 16)
scale=8.0               # default 8.0
lr=1e-2                 # learning rate (default 1e-2)
num_gpus=1              # number of gpus to use (default 1)

# experiment_name=${scene_name}_s${scale}_lr${lr}_dima${dim_a}_dimg${dim_g}_r${rep}_e${epochs}_b${batch_size}_d${downsample}_gpu${num_gpus}
experiment_name=counter_dark_s8.0_lr1e-2_dima48_dimg16_r5_e20_b8192_d1.0_gpu1
echo Experiment name : $experiment_name

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