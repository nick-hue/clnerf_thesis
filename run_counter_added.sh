#!/bin/bash
export DATA_DIR=/workspace/data/counter_sm_merged

scene=counter
rep=10
epochs=10
down=0.5
task_curr=1
task_number=2

exp_name=${scene}_r${rep}_e${epochs}_t${task_curr}
base_ckpt=ckpts/NGPGv2_CL/colmap_ngpa_CLNerf/${scene}_r${rep}_e${epochs}_t0/epoch=$((epochs-1))_slim.ckpt

echo $base_ckpt
# copy over replay‐buffer from the base run
mkdir -p results/NGPGv2/CLNerf/colmap_ngpa_CLNerf/${exp_name}/rep
cp results/NGPGv2/CLNerf/colmap_ngpa_CLNerf/${scene}_r${rep}_e${epochs}_t0/rep/rep_buf.torchSave \
   results/NGPGv2/CLNerf/colmap_ngpa_CLNerf/${exp_name}/rep/rep_buf.torchSave

python train_ngpgv2_CLNerf.py \
  --root_dir      ${DATA_DIR} \
  --dataset_name  colmap_ngpa_CLNerf \
  --exp_name      ${exp_name} \
  --num_epochs    ${epochs} \
  --batch_size    8192 \
  --lr            1e-2 \
  --rep_size      ${rep} \
  --eval_lpips \
  --task_curr     ${task_curr} \
  --task_number   ${task_number} \
  --weight_path   ${base_ckpt} \
  --dim_a         48 \
  --scale         8.0 \
  --downsample    ${down} \
  --vocab_size    ${task_number}
