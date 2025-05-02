export DATA_DIR=/workspace/data/counter_sm_merged/ # where counter dataset is saved

# Stage 2: continual (added) training
task_number=2
task_curr=1
scene_name=counter
rep=10
epochs=20
downsample=1.0

exp_base=${scene_name}_r${rep}_e${epochs}_t1
exp_add=${scene_name}_r${rep}_e${epochs}_t2

# path to the slimmed base ckpt you just made
BASE_CKPT=ckpts/NGPGv2_CL/colmap_ngpa_CLNerf/${exp_base}/epoch=${epochs-1}.ckpt

python train_ngpgv2_CLNerf.py \
  --root_dir    $DATA_DIR \
  --dataset_name  colmap_ngpa_CLNerf \
  --exp_name      ${exp_add} \
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
  --weight_path   ${BASE_CKPT}