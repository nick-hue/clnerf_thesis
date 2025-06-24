#!/usr/bin/env python3
import os
import glob
import time
import argparse
from tensorboard.backend.event_processing import event_accumulator

def load_event_file(event_file):
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    return ea

def pick_latest_event(version_dir):
    evs = sorted(glob.glob(os.path.join(version_dir, "events.out.tfevents*")))
    return evs[-1] if evs else None

def get_versions(exp_path, all_versions):
    versions = sorted(glob.glob(os.path.join(exp_path, "version_*")))
    return versions if all_versions else versions[-1:]

def format_duration(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def show_summary(base_dir, exp_prefix, all_versions):
    rows = []
    for exp_dir in sorted(os.listdir(base_dir)):
        if not exp_dir.startswith(exp_prefix):
            continue
        exp_path = os.path.join(base_dir, exp_dir)

        for version in get_versions(exp_path, all_versions):
            ev_file = pick_latest_event(version)
            if not ev_file:
                continue

            try:
                ea = load_event_file(ev_file)
            except Exception as e:
                print(f"⚠️  Failed to load {ev_file}: {e}")
                continue

            tags = ea.Tags().get("scalars", [])
            # skip if required tags not present
            if "train/psnr" not in tags or "train/loss" not in tags:
                continue

            ps_entries = ea.Scalars("train/psnr")
            ls_entries = ea.Scalars("train/loss")
            if not ps_entries or not ls_entries:
                continue

            # median of last 50 values
            last50_ps   = [e.value for e in ps_entries][-50:]
            last50_ls   = [e.value for e in ls_entries][-50:]
            median_ps   = sorted(last50_ps)[len(last50_ps)//2]
            median_ls   = sorted(last50_ls)[len(last50_ls)//2]

            # compute training duration
            start = ps_entries[0].wall_time
            end   = ps_entries[-1].wall_time
            duration = format_duration(end - start)

            rows.append((
                f"{exp_dir}/{os.path.basename(version)}",
                median_ps, median_ls, duration
            ))

    if not rows:
        print(f"No matching experiments under '{base_dir}' starting with '{exp_prefix}'")
        return

    # sort by PSNR descending
    rows.sort(key=lambda x: x[1], reverse=True)


    # print table
    print(f"\n{'Experiment (version)':75}  {'PSNR':>8s}   {'Loss':>8s}   {'Time':>8s}")
    print("-" * 110)
    for name, psnr, loss, ttime in rows:
        print(f"{name:75s}   {psnr:8.4f}   {loss:8.4f}   {ttime:>8s}")

def main():
    p = argparse.ArgumentParser(
        description="Summarize median PSNR/loss & training time from TensorBoard logs."
    )
    p.add_argument("experiment_prefix",
                   help="only process experiment dirs starting with this")
    p.add_argument("-b", "--base-dir",
                   default="logs/NGPGv2_CL/colmap_ngpa_CLNerf",
                   help="root directory containing versioned TensorBoard logs")
    p.add_argument("-a", "--all-versions", action="store_true",
                   help="include all version_* subdirs instead of only the latest")
    args = p.parse_args()

    show_summary(args.base_dir, args.experiment_prefix, args.all_versions)

if __name__ == "__main__":
    main()
