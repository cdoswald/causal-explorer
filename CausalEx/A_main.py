import json
import multiprocessing as mp
import os
import time

from config import RunArgs, ExperimentArgs
from A_multiple_experiments import run_all_experiments


def main():

    # Record start time
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')

    # Instantiate run arguments (applies to all experiments)
    run_args = RunArgs()
    run_args.setup_dirs()
    run_args.save_config(os.path.join(run_args.run_dir, "run_config.json"))

    # Load random seeds
    with open("seeds_list.json", "r") as io:
        seeds = json.load(io)[:run_args.use_n_seeds]

    # Run experiments
    run_all_experiments(run_args, seeds)

    # Record end time
    end_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Run start time: {start_time} \nRun end time: {end_time}")


if __name__ == "__main__":
    main()
