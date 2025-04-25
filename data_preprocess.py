import argparse
import time

from preprocess_TUSZ import preprocess_TUSZ
from preprocess_Siena import preprocess_Siena
from preprocess_CHB_MIT import preprocess_CHB_MIT


def main():
    parser = argparse.ArgumentParser(description="Preprocess seizure EEG dataset.")
    parser.add_argument('--dataset', default='Siena', help="dataset name.")
    args = parser.parse_args()

    start_time = time.time()
    if args.dataset == 'TUSZ':
        print("Processing TUSZ dataset...")
        preprocess_TUSZ()
    elif args.dataset == 'Siena':
        print("Processing Siena dataset...")
        preprocess_Siena()
    elif args.dataset == 'CHB-MIT':
        print("Processing CHB-MIT dataset...")
        preprocess_CHB_MIT()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print(f"Preprocessing completed in {hours:02d}:{minutes:02d}:{seconds:.2f} (hh:mm:ss).")


if __name__ == '__main__':
    main()