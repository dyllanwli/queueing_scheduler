import argparse
import time
import numpy as np

# Service rates (must match scheduler.py)
MU_CPU = 0.5
MU_PRE = 10.0
MU_GPU = 5.0
MU_POST = 10.0

def simulate_inference(stage):
    if stage == "cpu":
        service_time = np.random.exponential(1 / MU_CPU)
    elif stage == "pre":
        service_time = np.random.exponential(1 / MU_PRE)
    elif stage == "gpu":
        service_time = np.random.exponential(1 / MU_GPU)
    elif stage == "post":
        service_time = np.random.exponential(1 / MU_POST)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    time.sleep(service_time)
    print(f"Stage '{stage}' completed in {service_time:.2f}s")
    return service_time

def main():
    parser = argparse.ArgumentParser(description="Simulate LiDAR inference stages.")
    parser.add_argument('--stage', choices=['cpu', 'pre', 'gpu', 'post'], required=True,
                        help="Specify the processing stage: 'cpu' (CPU-only), 'pre', 'gpu', or 'post'.")
    args = parser.parse_args()
    
    simulate_inference(args.stage)

if __name__ == "__main__":
    main()