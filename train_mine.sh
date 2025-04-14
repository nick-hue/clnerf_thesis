export DATA_DIR=/workspace/data/
scene_name=breville
rep=10
task_curr=0
task_number=1
downsample=1.0

python train_ngpgv2_CLNerf.py \
      --root_dir $DATA_DIR --dataset_name colmap_ngpa_CLNerf \
      --exp_name ${scene_name}_${rep} \
      --num_epochs 20 --batch_size 8192 --lr 1e-2 --rep_size $rep --eval_lpips \
      --task_curr $task_curr --task_number $task_number --dim_a 48 --scale 8.0 --downsample ${downsample} --vocab_size ${task_number}