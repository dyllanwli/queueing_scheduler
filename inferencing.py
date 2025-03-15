import argparse
import time
# inference simulation
def main():
    parser = argparse.ArgumentParser(description="A fake model script.")
    parser.add_argument('--device', choices=['gpu', 'cpu'], required=True, 
                        help="Specify the device to use: 'gpu' or 'cpu'.")

    args = parser.parse_args()

    if args.device == 'gpu':
        print("Running on GPU...")
        time.sleep(1)
    elif args.device == 'cpu':
        print("Running on CPU...")
        time.sleep(10)

    print("Task completed.")

if __name__ == "__main__":
    main()