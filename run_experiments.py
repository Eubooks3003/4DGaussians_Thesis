import subprocess
import time
from queue import Queue
from threading import Thread
import os
import argparse
import pynvml

# ---- PARSE ARGUMENTS ----
parser = argparse.ArgumentParser(description="Parallel experiment launcher with GPU assignment")
parser.add_argument("--method", type=str, choices=["rand", "oracle", "AV"], required=True,
                    help="View selection method: rand, oracle, or AV")
args = parser.parse_args()
METHOD = args.method

# ---- CONFIG ----
NUM_EXPERIMENTS = 1           # total experiments to run
MAX_PARALLEL = 1               # how many to run in parallel
BASE_PORT = 6000               # starting port
DATASET_PATH = "/workspace/Dataset/aleks-teapot/"
EXPNAME = "hypernerf/aleks"
CONFIG_PATH = "arguments/hypernerf/default.py"

# ---- INIT NVML ----
pynvml.nvmlInit()
NUM_GPUS = pynvml.nvmlDeviceGetCount()

# ---- UTILITY: Get least loaded GPU ----
def get_least_used_gpu():
    min_used_mem = float("inf")
    selected_gpu = 0
    for i in range(NUM_GPUS):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = mem_info.used
        if used < min_used_mem:
            min_used_mem = used
            selected_gpu = i
    return selected_gpu

# ---- JOB FUNCTION ----
def run_experiment(exp_num, port):
    gpu_id = get_least_used_gpu()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python", "train.py",
        "-s", DATASET_PATH,
        "--port", str(port),
        "--expname", EXPNAME,
        "--configs", CONFIG_PATH,
        "--exp_num", str(exp_num),
        "--view_selection_method", METHOD
    ]
    print(f"Starting experiment {exp_num} on port {port} using GPU {gpu_id} with method {METHOD}")
    return subprocess.Popen(cmd, env=env)

# ---- WORKER THREAD ----
def worker(q, running_procs):
    while not q.empty():
        exp_num = q.get()
        port = BASE_PORT + exp_num
        proc = run_experiment(exp_num, port)
        running_procs.append((exp_num, proc))
        proc.wait()  # Block until it finishes
        print(f"Experiment {exp_num} finished.")
        running_procs.remove((exp_num, proc))
        q.task_done()

# ---- MAIN ----
if __name__ == "__main__":
    job_queue = Queue()
    for i in range(NUM_EXPERIMENTS):
        job_queue.put(i)

    running = []
    threads = []
    for _ in range(MAX_PARALLEL):
        t = Thread(target=worker, args=(job_queue, running))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("All experiments finished.")
