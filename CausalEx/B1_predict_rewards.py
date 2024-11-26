"""Compare success of Q-models in predicting rewards (given state and action pairs)
for causal vs. random information."""

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


def train_q_model(args):

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Set up environment
    env = gym.make(args.env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(args.seed)

    # Set up models
    qf1 = SoftQNetwork(env).to(device)
    qf1_optimizer = optim.Adam(qf1.parameters(), lr=args.q_lr)

    # Prepopulate replay buffer
    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    if args.cx_mode.lower() == "causal":
        rb = prepopulate_buffer_causal(env, rb, args)
    elif args.cx_mode.lower() == "random":
        rb = prepopulate_buffer_random(env, rb, args)
        
    # Train Q-model to predict reward conditional on state and action
    epoch_losses = []
    for epoch_i in range(args.n_epochs):
        batch_losses = []
        for _ in range(args.n_samples_per_epoch):
            data = rb.sample(args.batch_size)
            qf1_pred_rewards = qf1(data.observations, data.actions)
            qf1_loss = F.mse_loss(qf1_pred_rewards, data.rewards)
            qf1_optimizer.zero_grad()
            qf1_loss.backward()
            qf1_optimizer.step()
            batch_losses.append(qf1_loss.item())

        # Record avg epoch loss and print progress
        avg_epoch_loss = np.mean(batch_losses)
        epoch_losses.append(avg_epoch_loss)
        print(f"Completed epoch {epoch_i + 1} with avg loss {round(avg_epoch_loss, 5)}")

    # Save epoch losses to disk
    file_name = f"reward_epoch_losses_env_{args.env_id}_mode_{args.cx_mode}_buffer_{args.buffer_size}.json"
    with open(os.path.join(args.loss_data_dir, file_name), "w") as io:
        json.dump(epoch_losses, io)
        print(f"Successfully saved {file_name} file")


if __name__ == "__main__":

    # Set up multiprocessing
    num_cores = os.cpu_count()
    num_workers = 24 #int(num_cores * 3) // 4
    process_args = []

    # Set experiment parameters
    seed = 42 #TODO: average over seeds?
    env_ids = [
        "Ant-v4",
        "HalfCheetah-v4",
        # "Hopper-v4", #TODO: fix XML file
        "Humanoid-v4",
        # "Walker2d-v4", #TODO: fix XML file
    ]
    cx_modes = ["causal", "random"]
    test_buffer_sizes = [10_000, 100_000, 500_000, 1_000_000]

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
                    } for cx_mode in cx_modes
                } for buffer_size in test_buffer_sizes
            } for env_id in env_ids        
        })
        loss_data_dir = os.path.join("runs", "predict_rewards")
        os.makedirs(loss_data_dir, exist_ok=True)

        # Record start time
        start_time = time.strftime('%Y-%m-%d %H:%M:%S')

        # Test all environments
        for env_id in env_ids:

            # Test all buffer sizes
            for buffer_size in test_buffer_sizes:

                # Test cx modes (causal vs random)
                for cx_mode in cx_modes:

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
        with mp.Pool(processes=num_workers) as pool:
            pool.map(train_q_model, process_args)

        # Record end time and report progress
        end_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"Run start time: {start_time} \nRun end time: {end_time}")
