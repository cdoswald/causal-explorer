import json
import multiprocessing as mp
import os
import time

from CausalEx.config import RunArgs, ExperimentArgs
from CausalEx.A_run_single_experiment import run_single_experiment


def test_A_run_all_experiments():

    # Instantiate run arguments (applies to all experiments)
    run_args = RunArgs()
    run_args.run_name = f"regression_test_{time.strftime('%Y%m%d_%H%M%S')}"
    run_args.run_dir = os.path.join("runs", "testing", run_args.run_name)
    run_args.setup_dirs()

    run_args.use_n_seeds = 2

    # Load random seeds
    with open("../seeds_list.json", "r") as io:
        seeds = json.load(io)[:run_args.use_n_seeds]

    ## ------------------------------------------------------------------------
    ## Unfortunately, have to reproduce most of the "run_all_experiments" function
    ## here to be able to change the number of timesteps and other experiment arguments
    ## ------------------------------------------------------------------------
    process_args = []
    for env_id in run_args.env_ids:
        for cx_mode in run_args.cx_modes:
            for seed in seeds:
                exp_args = ExperimentArgs(
                    env_id=env_id,
                    cx_mode=cx_mode,
                    seed=seed,
                )
                exp_args.create_exp_dir()

                # Modify experiment args to reduce time for testing
                exp_args.train_timesteps = 1000
                exp_args.eval_timesteps = 100
                exp_args.prepopulate_buffer_hard_cap = 1000
                exp_args.max_steps_per_interact = 10

                # Add experiment-specific arguments to processes list
                process_args.append(exp_args)

    # Start processes
    with mp.Pool(processes=run_args.num_workers) as pool:
        pool.map(run_single_experiment, process_args)
    
    # Clean up directories
    #TODO