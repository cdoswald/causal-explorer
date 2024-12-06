import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer

from causal_explorer import prepopulate_buffer_causal, prepopulate_buffer_random
from config import RunArgs, ExperimentArgs
from models import SoftQNetwork


def main():

    # Instantiate run arguments (applies to all experiments)
    run_args = RunArgs()

    # Set experiment parameters
    seed = 42 #TODO: average over seeds?
    test_buffer_sizes = [5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

    # Set training parameters
    n_epochs = 20
    n_samples_per_epoch = 5000

    # Use shared loss data dict
    with mp.Manager() as manager:

        # Set up loss data dict
        loss_data = manager.dict({
            env_id:{
                buffer_size:{
                    cx_mode:{
                        epoch_i:[] for epoch_i in range(n_epochs)
                    } for cx_mode in run_args.cx_modes
                } for buffer_size in test_buffer_sizes
            } for env_id in run_args.env_ids
        })
        loss_data_dir = os.path.join("runs", "predict_rewards")
        os.makedirs(loss_data_dir, exist_ok=True)

        # Record start time
        start_time = time.strftime('%Y-%m-%d %H:%M:%S')

        # Test all environments
        process_args = []
        for env_id in run_args.env_ids:

            # Test all buffer sizes
            for buffer_size in test_buffer_sizes:

                # Test cx modes (causal vs random)
                for cx_mode in run_args.cx_modes:

                    # Create experiment arguments
                    exp_args = ExperimentArgs(
                        env_id=env_id,
                        cx_mode=cx_mode,
                        buffer_size=buffer_size,
                        seed=seed,
                    )
                    exp_args.n_epochs = n_epochs
                    exp_args.n_samples_per_epoch = n_samples_per_epoch
                    exp_args.loss_data_dir = loss_data_dir

                    # Add experiment-specific arguments to processes list
                    process_args.append((exp_args))

        # Start processes
        with mp.Pool(processes=run_args.num_workers) as pool:
            pool.map(train_q_model, process_args)

        # Record end time and report progress
        end_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"Run start time: {start_time} \nRun end time: {end_time}")
