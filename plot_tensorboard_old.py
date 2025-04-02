import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict

ROOT_DIR = "/workspace/4DGaussians_Thesis/output"
SUBDIR_SUFFIX = "hypernerf/aleks"
OUTPUT_DIR = "tb_overlay_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Select the folders you want to include
SELECTED_FOLDERS = [
    "output_AVS_1",
    "output_random_0",
    "output_random_1",
    "output_random_2",
    "output_random_3",
    "output_random_4",
    "output_random_10",
    "output_random_12",
    "output_random_13",
    "output_AVS_2",
    "output_AVS_10"
]

def extract_group_name(folder_name):
    match = re.match(r"(output_[a-zA-Z]+)", folder_name)
    return match.group(1) if match else folder_name

def load_scalars(event_dir):
    ea = event_accumulator.EventAccumulator(event_dir)
    try:
        ea.Reload()
    except Exception as e:
        print(f"⚠️ Skipping {event_dir}: {e}")
        return {}
    scalars = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        scalars[tag] = (steps, values)
    return scalars

def align_series(series_list):
    """Interpolates all series to the smallest common step range"""
    common_steps = set(series_list[0][0])
    for steps, _ in series_list[1:]:
        common_steps &= set(steps)
    if not common_steps:
        return None, None
    common_steps = sorted(common_steps)
    interpolated = []
    for steps, values in series_list:
        interp_vals = np.interp(common_steps, steps, values)
        interpolated.append(interp_vals)
    return np.array(common_steps), np.vstack(interpolated)

def plot_all_metrics(folders):
    grouped_data = defaultdict(lambda: defaultdict(list))

    for folder in folders:
        group = extract_group_name(folder)
        full_path = os.path.join(ROOT_DIR, folder, SUBDIR_SUFFIX)
        scalars = load_scalars(full_path)
        for tag, (steps, values) in scalars.items():
            grouped_data[tag][group].append((steps, values))

    for tag, group_runs in grouped_data.items():
        plt.figure(figsize=(12, 6))  # Wider for slides
        random_std = None
        aligned_steps = None

        if "output_random" in group_runs:
            aligned_steps, random_vals = align_series(group_runs["output_random"])
            if aligned_steps is None:
                print(f"⚠️ Skipping '{tag}' — no overlap in random group.")
                continue
            random_std = np.std(random_vals, axis=0)
            print(f"ℹ️ Using random std for tag '{tag}'")

        color_cycle = plt.cm.tab10.colors  # Colorblind-friendly
        plt.rcParams.update({
            'font.size': 16,
            'axes.labelsize': 18,
            'axes.titlesize': 20,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14
        })

        for idx, (group_name, runs) in enumerate(group_runs.items()):
            aligned_x, aligned_y = align_series(runs)
            if aligned_x is None:
                print(f"⚠️ Skipping group '{group_name}' for tag '{tag}': no shared steps.")
                continue
            mean = np.mean(aligned_y, axis=0)

            # Plot with either shared or group-specific error
            if group_name == "output_random":
                std = np.std(aligned_y, axis=0)
            elif random_std is not None and aligned_steps is not None:
                std = np.interp(aligned_x, aligned_steps, random_std)
            else:
                std = np.zeros_like(mean)

            plt.plot(
                aligned_x, mean,
                label=group_name.replace("output_", "").upper(),
                linewidth=2.5,
                color=color_cycle[idx % len(color_cycle)]
            )
            plt.fill_between(
                aligned_x, mean - std, mean + std,
                alpha=0.2,
                color=color_cycle[idx % len(color_cycle)]
            )

        plt.title(tag.replace("_", " ").upper())
        plt.xlabel("Training Step")
        plt.ylabel("Metric Value")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        safe_tag = tag.replace("/", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{safe_tag}.png"), dpi=300)
        plt.close()
        print(f"📈 Saved thesis-ready plot for '{tag}'")


if __name__ == "__main__":
    plot_all_metrics(SELECTED_FOLDERS)
