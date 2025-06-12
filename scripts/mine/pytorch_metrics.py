import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import time

def get_training_time(ea):
    start_time = ea.Scalars("train/psnr")[0].wall_time
    end_time = ea.Scalars("train/psnr")[-1].wall_time
    duration = end_time - start_time
    duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))

    return duration_str

def show_experiment(exp_name):
    for exp_dir in os.listdir(base_dir):
        if not exp_dir.startswith(exp_name):
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
        break

    # for e in ea.Scalars("train/psnr"):
    #     print(f"Step: {e.step}, PSNR: {e.value}, Wall time: {e.wall_time}")
    # difference in wall time
    training_time = get_training_time(ea) 
    
    print(f"Experiment: {exp_name}, Training Time: {training_time}")


def get_all_experiments_data_prefix(prefix, base_dir="logs/NGPGv2_CL/colmap_ngpa_CLNerf"):

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
        
        t_time = "N/A"
        try:
            t_time = get_training_time(ea)
        except Exception as e:
            print(f"Failed to get training time for {exp_dir}: {e}")    

        results.append((exp_dir, median_psnr, median_loss, t_time))

    return results

base_dir        = "logs/NGPGv2_CL/colmap_ngpa_CLNerf"
experiment_pref = "counter_shirt"

results = get_all_experiments_data_prefix(prefix=experiment_pref, base_dir=base_dir)

# sort by PSNR descending
results.sort(key=lambda x: x[1], reverse=True)

# ——— 3``. Print final table ———
print(f"\n{'Experiment':70s}  {'PSNR':>8s}   {'Loss':>8s}    {'Train Time':>10s}")
print("-"*105)
for name, psnr, loss, ttime in results:
    print(f"{name:70s}   {psnr:8.4f}   {loss:8.4f}   {ttime:>10s}")

# show_experiment(exp_name="counter_shirt_high")