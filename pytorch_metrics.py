import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import time

table = '''
Experiment Name	Scale	Learning Rate	dim_a Size	dim_g Size	Epochs	Batch Size	Downsampling	#GPU	Training Time	Testing Rendering	Total Time
drz_s8.0_lr1e-2_dima48_dimg16_r5_e10_b4096_d1.0_gpu1	8.0	0.01	48	16	10	4096	1.0	1	37:43	17:51	55m34s
drz_s16.0_dima48_dimg16_r5_e10_b4096_d1.0_gpu1	16.0	0.01	48	16	10	4096	1.0	1	24:26	22:12	46m38s
drz_s32.0_dima48_dimg16_r5_e10_b4096_d1.0_gpu1	32.0	0.01	48	16	10	4096	1.0	1	19:41	23:56	43m37s
drz_s48.0_dima48_dimg16_r5_e10_b4096_d1.0_gpu1	48.0	0.01	48	16	10	4096	1.0	1	20:01	20:14	40m15s
drz_s64.0_dima48_dimg16_r5_e10_b4096_d1.0_gpu1	64.0	0.01	48	16	10	4096	1.0	1	20:54	21:26	42m40s
drz_s32.0_lr5e-3_dima48_dimg16_r5_e10_b4096_d1.0_gpu1	32.0	0.005	48	16	10	4096	1.0	1	17:29	21:25	38m54s
drz_s32.0_lr1e-2_dima48_dimg16_r5_e20_b4096_d1.0_gpu1	32.0	0.01	48	16	20	4096	1.0	1	34:52	None	34m52s
drz_s32.0_lr5e-3_dima48_dimg16_r5_e20_b4096_d1.0_gpu1	32.0	0.005	48	16	20	4096	1.0	1	1:32:40	None	92m40s
drz_s32.0_lr1e-2_dima64_dimg32_r5_e20_b4096_d1.0_gpu1	32.0	0.01	64	32	20	4096	1.0	1	40:52	None	40m52s
drz_s32.0_lr1e-2_dima96_dimg32_r5_e20_b4096_d1.0_gpu1	32.0	0.01	64	32	20	4096	1.0	1	33:50	None	33m50s
drz_s32.0_lr1e-2_dima48_dimg24_r5_e10_b4096_d1.0_gpu1	32.0	0.01	48	24	10	4096	1.0	1	21:50	None	21m50s
drz_s8.0_lr1e-2_dima48_dimg16_r5_e10_b8192_d1.0_gpu1	8.0	0.01	48	16	10	8192	1.0	1	50:32	None	50m32s
drz_same_image_s32.0_lr1e-2_dima48_dimg16_r5_e10_b4096_d1.0_gpu1	32.0	0.01	48	24	10	8192	1.0	1	20:53	None	20m53s
drz_s32.0_lr2e-2_dima48_dimg16_r5_e10_b4096_d1.0_gpu1	32.0	0.02	48	16	10	4096	1.0	1	32:58	None	32m58s
drz_same_image_s32.0_lr2e-2_dima64_dimg32_r5_e10_b4096_d1.0_gpu1 	32.0	0.02	64	32	10	4096	1.0	1	48:23	None	48m23s
'''

time_format_hours = "%H:%M:%S"
time_format_mins = "%M:%S"

training_times = {}
for line in table.strip().splitlines()[1:]:
    parts = line.split('\t')
    name, train_time = parts[0], parts[9]
    training_times[name] = train_time


base_dir        = "logs/NGPGv2_CL/colmap_ngpa_CLNerf"
experiment_pref = "counter"

results = []
for exp_dir in os.listdir(base_dir):
    if not exp_dir.startswith(experiment_pref):
        continue

    # pick the latest version folder
    versions = sorted(glob.glob(os.path.join(base_dir, exp_dir, "version_*")))
    if not versions:
        continue
    version = versions[-1]

    # pick the newest events file
    evs = sorted(glob.glob(os.path.join(version, "events.out.tfevents*")))
    if not evs:
        continue
    event_file = evs[-1]

    try:
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()
    except Exception as e:
        print(f"Failed to load {event_file}: {e}")
        continue

    tags = ea.Tags()["scalars"]
    if "train/psnr" not in tags or "train/loss" not in tags:
        continue

    # median PSNR
    last_n   = 50
    psnrs    = [e.value for e in ea.Scalars("train/psnr")[-last_n:]]
    median_psnr = sorted(psnrs)[len(psnrs)//2] if psnrs else float('nan')

    # median loss
    losses   = [e.value for e in ea.Scalars("train/loss")[-last_n:]]
    median_loss = sorted(losses)[len(losses)//2] if losses else float('nan')

    # lookup training time (or N/A)
    
    t_time = training_times.get(exp_dir, "N/A")
    if t_time != "N/A":
        # if it is in hours format
        try:
            time_obj = time.strptime(t_time, time_format_hours)
            t_time = time.strftime("%H:%M:%S", time_obj)
        # if it is in minutes format
        except ValueError:
            try:
                time_obj = time.strptime(t_time, time_format_mins)
                t_time = time.strftime("%H:%M:%S", time_obj)
            except ValueError:
                t_time = "N/A"

    results.append((exp_dir, median_psnr, median_loss, t_time))

# sort by PSNR descending
results.sort(key=lambda x: x[1], reverse=True)

# ——— 3. Print final table ———
print(f"\n{'Experiment':70s}  {'PSNR':>8s}   {'Loss':>8s}    {'Train Time':>10s}")
print("-"*105)
for name, psnr, loss, ttime in results:
    print(f"{name:70s}   {psnr:8.4f}   {loss:8.4f}   {ttime:>10s}")
