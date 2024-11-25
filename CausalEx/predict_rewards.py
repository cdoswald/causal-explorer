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
from models import Actor, SoftQNetwork

if __name__ == "__main__":

    # Define params
    seed = 9745

    n_epochs = 2
    n_samples_per_epoch = 1000

    args = ExperimentArgs()
    args.seed = seed
    test_buffer_sizes = [10_000, 100_000, 500_000, 1_000_000]

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up loss data dict
    loss_data = {
        env_id:{
            buffer_size:{
                cx_mode:{
                    epoch_i:[] for epoch_i in range(n_epochs)
                } for cx_mode in args.cx_modes
            } for buffer_size in test_buffer_sizes
        } for env_id in args.env_ids        
    }

    # Test all environments
    for env_id in args.env_ids:

        # Set up environment
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        # Set up models
        qf1 = SoftQNetwork(env).to(device)
        qf1_optimizer = optim.Adam(qf1.parameters(), lr=args.q_lr)

        # Test all buffer sizes
        for buffer_size in test_buffer_sizes:

            # Test cx modes (causal vs random)
            for cx_mode in args.cx_modes:

                # Prepopulate replay buffer
                env.observation_space.dtype = np.float32
                rb = ReplayBuffer(
                    buffer_size,
                    env.observation_space,
                    env.action_space,
                    device,
                    handle_timeout_termination=False,
                )
                if cx_mode.lower() == "causal":
                    rb = prepopulate_buffer_causal(env, rb, args)
                elif cx_mode.lower() == "random":
                    rb = prepopulate_buffer_random(env, rb, args)

                # Train Q-model to predict reward conditional on state and action
                loss_data = {epoch_i:[] for epoch_i in range(n_epochs)}
                for epoch_i in range(n_epochs):
                    epoch_losses = []
                    for _ in range(n_samples_per_epoch):
                        data = rb.sample(args.batch_size)

                        qf1_pred_rewards = qf1(data.observations, data.actions)
                        qf1_loss = F.mse_loss(qf1_pred_rewards, data.rewards)
                        qf1_optimizer.zero_grad()
                        qf1_loss.backward()
                        qf1_optimizer.step()

                        epoch_losses.append(qf1_loss.item())
                    
                    print(
                        f"Completed epoch {epoch_i + 1} " +
                        f"with avg loss {round(np.mean(epoch_losses), 5)}"
                    )
                        


    # 11/15 @ 12:30pm left off here; need to save loss data and then visualize
    # loss_data = {
    #     env_id:{
    #         buffer_size:{
    #             cx_mode:{
    #                 epoch_i:[] for epoch_i in range(n_epochs)
    #             } for cx_mode in args.cx_modes
    #         } for buffer_size in test_buffer_sizes
    #     } for env_id in args.env_ids        
    # }

