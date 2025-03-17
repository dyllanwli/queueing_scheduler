import numpy as np
from scipy.optimize import minimize_scalar
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Queueing model parameters (tune based on your system)
LAMBDA = 1.0      # Arrival rate (tasks/sec)
MU_CPU = 0.5      # CPU-only service rate (tasks/sec)
MU_PRE = 10.0     # Preprocessing rate (tasks/sec)
MU_GPU = 5.0      # GPU inference rate (tasks/sec)
MU_POST = 10.0    # Postprocessing rate (tasks/sec)

# Compute expected response time E[T] based on queueing model
def compute_expected_response_time(p, lambda_=LAMBDA, mu_cpu=MU_CPU, mu_pre=MU_PRE, mu_gpu=MU_GPU, mu_post=MU_POST):
    if p < 0 or p > 1:
        return np.inf
    
    # CPU queue: handles CPU-only tasks, preprocessing, and postprocessing
    lambda_cpu = lambda_ * (1 + p)  # (1-p) for CPU-only + p for pre + p for post
    e_s = ((1 - p) / (1 + p)) * (1 / mu_cpu) + (p / (1 + p)) * (1 / mu_pre) + (p / (1 + p)) * (1 / mu_post)
    e_s2 = ((1 - p) / (1 + p)) * 2 / (mu_cpu ** 2) + (p / (1 + p)) * 2 / (mu_pre ** 2) + (p / (1 + p)) * 2 / (mu_post ** 2)
    rho_cpu = lambda_cpu * e_s
    if rho_cpu >= 1:
        return np.inf
    w_cpu = (lambda_cpu * e_s2) / (2 * (1 - rho_cpu))
    
    # GPU queue: handles inference only
    rho_gpu = lambda_ * p / mu_gpu
    if rho_gpu >= 1:
        return np.inf
    w_gpu = rho_gpu / (mu_gpu * (1 - rho_gpu))
    
    # Response times
    e_t_cpu_only = w_cpu + 1 / mu_cpu
    e_t_gpu_path = 2 * w_cpu + w_gpu + 1 / mu_pre + 1 / mu_gpu + 1 / mu_post
    
    # Overall E[T]
    e_t = (1 - p) * e_t_cpu_only + p * e_t_gpu_path
    return e_t

# Find optimal p to minimize E[T]
def find_optimal_p():
    result = minimize_scalar(compute_expected_response_time, bounds=(0, 1), method='bounded')
    if result.success:
        return result.x, result.fun
    else:
        raise ValueError("Failed to optimize p")

# Task times dictionary: task_id -> [start_time, end_time]
task_times = {}
times_lock = threading.Lock()

# Set end time for a task
def set_end_time(task_id):
    with times_lock:
        task_times[task_id][1] = time.time()

# Execute a task stage using inference.py
def process_task(task_id, stage):
    cmd = ["python", "inference.py", "--stage", stage]
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    if result.returncode == 0:
        print(f"Task {task_id} stage '{stage}' completed in {duration:.2f}s")
    else:
        print(f"Task {task_id} stage '{stage}' failed: {result.stderr}")
    return duration

# Submit a task to CPU or GPU pipeline
def submit_task(task_id, mode, cpu_executor, gpu_executor):
    with times_lock:
        task_times[task_id] = [time.time(), None]
    
    if mode == "cpu":
        future = cpu_executor.submit(process_task, task_id, "cpu")
        future.add_done_callback(lambda f: set_end_time(task_id))
    elif mode == "gpu":
        def pre_callback(future):
            if future.exception() is None:
                inference_future = gpu_executor.submit(process_task, task_id, "gpu")
                inference_future.add_done_callback(post_callback)
        
        def post_callback(future):
            if future.exception() is None:
                post_future = cpu_executor.submit(process_task, task_id, "post")
                post_future.add_done_callback(lambda f: set_end_time(task_id))
        
        pre_future = cpu_executor.submit(process_task, task_id, "pre")
        pre_future.add_done_callback(pre_callback)

# Run the scheduler simulation
def run_scheduler(num_tasks=10, simulation_duration=10.0):
    global task_times
    task_times = {}
    
    # Compute optimal p
    optimal_p, min_et = find_optimal_p()
    print(f"Optimal p: {optimal_p:.3f}, Theoretical E[T]: {min_et:.3f}s")
    
    # Generate Poisson arrivals
    inter_arrival_times = np.random.exponential(1 / LAMBDA, num_tasks)
    arrival_times = np.cumsum(inter_arrival_times)
    tasks = [(i, t) for i, t in enumerate(arrival_times) if t < simulation_duration]
    print(f"Simulating {len(tasks)} tasks over {simulation_duration}s")
    
    # Executors for CPU and GPU (1 worker each to simulate single servers)
    with ThreadPoolExecutor(max_workers=1) as cpu_executor, ThreadPoolExecutor(max_workers=1) as gpu_executor:
        start_time = time.time()
        for task_id, arrival_time in tasks:
            # Wait until the task's arrival time
            current_time = time.time() - start_time
            if current_time < arrival_time:
                time.sleep(arrival_time - current_time)
            
            # Assign task to CPU or GPU path based on optimal_p
            mode = "gpu" if np.random.random() < optimal_p else "cpu"
            print(f"Task {task_id} arriving at {arrival_time:.2f}s, assigned to {mode.upper()}")
            submit_task(task_id, mode, cpu_executor, gpu_executor)
        
        # Wait for all tasks to complete
        while any(t[1] is None for t in task_times.values()):
            time.sleep(0.1)
    
    # Calculate and report response times
    response_times = [t[1] - t[0] for t in task_times.values() if t[1] is not None]
    avg_response_time = np.mean(response_times)
    print(f"\nSimulation completed in {time.time() - start_time:.2f}s")
    print(f"Average response time: {avg_response_time:.2f}s")
    print(f"CPU utilization estimate: {LAMBDA * (1 + optimal_p) / MU_CPU:.2f}")
    print(f"GPU utilization estimate: {LAMBDA * optimal_p / MU_GPU:.2f}")

if __name__ == "__main__":
    run_scheduler()