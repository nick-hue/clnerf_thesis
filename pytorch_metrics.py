import os
import glob
from tensorboard.backend.event_processing import event_accumulator

base_dir = "logs/NGPGv2_CL/colmap_ngpa_CLNerf"
experiment_prefix = "drz"  # whatever your experiment name starts with

for exp_dir in os.listdir(base_dir):
    if not exp_dir.startswith(experiment_prefix):
        continue

    # find any version_* subfolder
    version_paths = glob.glob(os.path.join(base_dir, exp_dir, "version_*"))
    if not version_paths:
        print(f"  ⚠️  No version_*/ subdir under {exp_dir}, skipping")
        continue

    # within each version, find all events.out.tfevents files
    for version_path in version_paths:
        ev_files = glob.glob(os.path.join(version_path, "events.out.tfevents*"))
        if not ev_files:
            print(f"  ⚠️  No events files under {version_path}")
            continue

        # you can pick the most recent one, or iterate through all
        ev_files.sort()  # lexicographic sort; if you want the newest:
        event_file = ev_files[-1]

        print(f"\n=== Loading {event_file} ===")
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()

        # list available scalar tags
        tags = ea.Tags()["scalars"]
        if "train/psnr" not in tags:
            print(f"  • train/psnr not logged in {exp_dir} @ {version_path}")
            continue

        # pull out the PSNR series
        psnr_events = ea.Scalars("train/psnr")
        print(f"Metrics for experiment {exp_dir}, version {os.path.basename(version_path)}:")
        # for ev in psnr_events[::20]:
        #     print(f"  step {ev.step:5d} → PSNR = {ev.value:.4f}")

        # print last    PSNR value
        if psnr_events:
            last_psnr = psnr_events[-1].value
            print(f"  Last PSNR value: {last_psnr:.4f}")
        else:
            print("  No PSNR events found.")