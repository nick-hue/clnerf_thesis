import os
import glob
from tensorboard.backend.event_processing import event_accumulator

base_dir = "logs/NGPGv2_CL/colmap_ngpa_CLNerf"
experiment_prefix = "drz"  # whatever your experiment name starts with

results = []

for exp_dir in os.listdir(base_dir):
    if not exp_dir.startswith(experiment_prefix):
        continue

    version_paths = glob.glob(os.path.join(base_dir, exp_dir, "version_*"))
    if not version_paths:
        print(f"  ⚠️  No version_*/ subdir under {exp_dir}, skipping")
        continue

    version_paths.sort()
    version_path = version_paths[-1]

    ev_files = glob.glob(os.path.join(version_path, "events.out.tfevents*"))
    if not ev_files:
        print(f"  ⚠️  No events files under {version_path}")
        continue

    ev_files.sort()
    event_file = ev_files[-1]

    try:
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()
    except Exception as e:
        print(f"  ❌  Failed to load {event_file}: {e}")
        continue

    tags = ea.Tags()["scalars"]
    if "train/psnr" not in tags or "train/loss" not in tags:
        print(f"  ⚠️  Missing train/psnr or train/loss in {event_file}")
        continue

    # pull PSNR
    psnr_events = ea.Scalars("train/psnr")
    last_n = 50
    psnr_vals = [ev.value for ev in psnr_events[-last_n:]]
    median_psnr = sorted(psnr_vals)[len(psnr_vals)//2] if psnr_vals else float('nan')

    # pull loss
    loss_events = ea.Scalars("train/loss")
    loss_vals = [ev.value for ev in loss_events[-last_n:]]
    median_loss = sorted(loss_vals)[len(loss_vals)//2] if loss_vals else float('nan')

    results.append((exp_dir, median_psnr, median_loss))

# sort by PSNR descending
results.sort(key=lambda x: x[1], reverse=True)

print("\n=== Experiments sorted by median PSNR (last 50 steps) ===")
print(f"{'Experiment':60s}  {'PSNR':>8s}   {'Loss':>8s}")
print("-"*85)
for exp_dir, med_psnr, med_loss in results:
    print(f"{exp_dir:60s}   {med_psnr:8.4f}   {med_loss:8.4f}")
