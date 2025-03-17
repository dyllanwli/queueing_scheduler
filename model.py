import time
import argparse

def run_model(device='cpu'):
    """
    Simulate model running on different devices
    Args:
        device (str): 'cpu' or 'gpu'
    """
    print(f"Running model on {device.upper()}...")
    
    if device.lower() == 'cpu':
        time.sleep(3)  # Simulate CPU processing time
    elif device.lower() == 'gpu':
        time.sleep(1)  # Simulate GPU processing time
    else:
        raise ValueError("Device must be either 'cpu' or 'gpu'")
    
    print(f"Finished processing on {device.upper()}")
    return f"Model results from {device.upper()}"

def main():
    parser = argparse.ArgumentParser(description='Simulate model processing time')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], 
                      default='cpu', help='Device to run the model on')
    
    args = parser.parse_args()
    result = run_model(args.device)
    print(result)

if __name__ == "__main__":
    main()