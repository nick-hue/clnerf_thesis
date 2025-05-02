#!/bin/bash

# export DATA_DIR=/workspace/data/counter_sm_merged/ # where counter dataset is saved
export DATA_DIR=/workspace/data/counter_sm_added_p1/ # part one added dataset

task_curr=1
task_number=2
scene_name=counter
rep=10
epochs=20
downsample=1.0

exp_base=${scene_name}_r${rep}_e${epochs}_t$task_curr
exp_added=${scene_name}_r${rep}_e${epochs}_t$task_number

base_ckpt=ckpts/NGPGv2_CL/colmap_ngpa_CLNerf/${exp_base}/epoch=19.ckpt
echo Base experiment name   : $exp_base
echo Base checkpoint path   : $base_ckpt
echo Added experiment name  : $exp_added

mkdir -p results/NGPGv2/CLNerf/colmap_ngpa_CLNerf/${exp_added}/rep/
cp results/NGPGv2/CLNerf/colmap_ngpa_CLNerf/${exp_base}/rep/rep_buf.torchSave results/NGPGv2/CLNerf/colmap_ngpa_CLNerf/${exp_added}/rep/rep_buf.torchSave

python train_ngpgv2_CLNerf.py \
  --root_dir    $DATA_DIR \
  --dataset_name  colmap_ngpa_CLNerf \
  --exp_name      ${exp_added} \
  --num_epochs    ${epochs} \
  --batch_size    8192 \
  --lr            1e-2 \
  --rep_size      ${rep} \
  --eval_lpips \
  --task_curr     ${task_curr} \
  --task_number   ${task_number} \
  --dim_a         48 \
  --scale         8.0 \
  --downsample    ${downsample} \
  --weight_path   ${base_ckpt}