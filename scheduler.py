# scheduler.py
import numpy as np
from scipy.optimize import minimize_scalar
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Queueing model parameters (tune these based on your system)
LAMBDA = 1.0      # Arrival rate (frames/sec)
MU_CPU = 0.5      # CPU-only service rate (tasks/sec)
MU_PRE = 10.0     # Preprocessing rate (tasks/sec)
MU_GPU = 5.0      # GPU inference rate (tasks/sec)
MU_POST = 10.0    # Postprocessing rate (tasks/sec)

# Function to compute E[T] based on queueing model
def compute_expected_response_time(p, lambda_=LAMBDA, mu_cpu=MU_CPU, mu_pre=MU_PRE, mu_gpu=MU_GPU, mu_post=MU_POST):
    if p < 0 or p > 1:
        return np.inf
    
    # CPU queue
    lambda_cpu = lambda_ * (1 + p)
    e_s = ((1 - p) / (1 + p)) * (1 / mu_cpu) + (p / (1 + p)) * (1 / mu_pre) + (p / (1 + p)) * (1 / mu_post)
    e_s2 = ((1 - p) / (1 + p)) * 2 / (mu_cpu ** 2) + (p / (1 + p)) * 2 / (mu_pre ** 2) + (p / (1 + p)) * 2 / (mu_post ** 2)
    rho_cpu = lambda_cpu * e_s
    if rho_cpu >= 1:
        return np.inf
    w_cpu = (lambda_cpu * e_s2) / (2 * (1 - rho_cpu))
    
    # GPU queue
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

# Optimize p (could use GA, but here we use a simpler method for demo)
def find_optimal_p():
    result = minimize_scalar(compute_expected_response_time, bounds=(0, 1), method='bounded')
    if result.success:
        return result.x, result.fun
    else:
        raise ValueError("Failed to optimize p")

# Task execution function
def execute_task(task_id, mode):
    cmd = ["python", "inferencing.py", mode]
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    print(f"Task {task_id} ({mode}): {result.stdout.strip()}, Duration: {duration:.2f}s")
    return duration

# Scheduler with simulation
def run_scheduler(num_tasks=10, simulation_duration=10.0):
    optimal_p, min_et = find_optimal_p()
    print(f"Optimal p: {optimal_p:.3f}, Estimated E[T]: {min_et:.3f}s")
    
    # Simulate task arrivals (Poisson process)
    inter_arrival_times = np.random.exponential(1 / LAMBDA, num_tasks)
    arrival_times = np.cumsum(inter_arrival_times)
    tasks = [(i, t) for i, t in enumerate(arrival_times) if t < simulation_duration]
    
    # Thread-safe queues for simulation
    cpu_queue = []
    gpu_queue = []
    queue_lock = threading.Lock()
    
    # Assign tasks based on optimal p
    def schedule_task(task_id, arrival_time):
        with queue_lock:
            if np.random.random() < optimal_p:
                mode = "gpu"
                gpu_queue.append((task_id, arrival_time))
                print(f"Task {task_id} assigned to CPU-GPU pipeline at {arrival_time:.2f}s")
            else:
                mode = "cpu"
                cpu_queue.append((task_id, arrival_time))
                print(f"Task {task_id} assigned to CPU-only at {arrival_time:.2f}s")
        return mode
    
    # Process tasks in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:  # One worker per resource
        futures = []
        for task_id, arrival_time in tasks:
            time.sleep(max(0, arrival_time - (time.time() - start_time)))  # Wait until arrival
            mode = schedule_task(task_id, arrival_time)
            futures.append(executor.submit(execute_task, task_id, mode))
        
        # Collect results
        durations = [f.result() for f in futures]
    
    avg_duration = np.mean(durations)
    print(f"Average task duration: {avg_duration:.2f}s")

if __name__ == "__main__":
    start_time = time.time()
    run_scheduler()
    print(f"Total simulation time: {time.time() - start_time:.2f}s")