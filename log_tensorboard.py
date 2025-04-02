import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def export_tensorboard_scalars(log_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading TensorBoard logs from: {log_dir}")
    ea = EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    print(f"Found scalar tags: {tags}")

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # Plot and save
        plt.figure()
        plt.plot(steps, values, label=tag)
        plt.xlabel('Step')
        plt.ylabel(tag.split('/')[-1])
        plt.title(tag)
        plt.legend()
        plt.tight_layout()

        safe_tag = tag.replace('/', '_')
        out_path = os.path.join(output_dir, f"{safe_tag}.png")
        plt.savefig(out_path)
        plt.close()

        print(f"Saved: {out_path}")

if __name__ == "__main__":
    log_dir = "./output_AVS_0/hypernerf/aleks"  # <- path to TensorBoard event files
    output_dir = "./tensorboard_plots"          # <- where PNGs will go

    export_tensorboard_scalars(log_dir, output_dir)
