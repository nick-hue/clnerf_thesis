#!/bin/bash

# render video using CLNeRF model
scene=breville
task_number=2
task_curr=0
rep=20
scale=8.0
render_fname=render_test_breville_meow

ckpt_path=/workspace/CLNeRF/CLNeRF/ckpts/NGPGv2_CL/colmap_ngpa_CLNerf/breville_r20_e20/epoch=19.ckpt

# bash scripts/CLNeRF/WAT/render_video.sh $task_number $task_curr $scene $ckpt_path $rep $scale $render_fname
bash render_video_mine.sh $task_number $task_curr $scene $ckpt_path $rep $scale $render_fname
