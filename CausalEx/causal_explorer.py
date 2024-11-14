from itertools import combinations
from math import comb
import random
import time

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import torch

# TODO: think about how to set random seed between action samples
# TODO: benchmark speed and success of causal explorer vs random exploration
# TODO: plot histogram of rewards of different n-way interactions
# TODO: render trajectories of causal explorer
# TODO: consider slight perturbations (positive and negative) relative to baseline action
# TODO: explore how batch size and training frequency params affect average episode return
# TODO: explore diminishing returns of causal exploration period length

ADJUST_SEED = 829 # don't use exact same seed as main RL pipeline


def prepopulate_buffer_causal(env, rb, args) -> ReplayBuffer:
    """Runs Causal Explorer method to prepopulate replay buffer
    with experimental/controlled data.
    
    Arguments
        env: instantiated gymnasium environment
        rb: instantiated stable_baselines3 replay buffer
        args: configuration arguments dataclass
    
    Returns
        stable_baselines3 replay buffer with trajectory data
    """
    # set up random seeds and device
    random.seed(args.seed + ADJUST_SEED)
    np.random.seed(args.seed + ADJUST_SEED)
    torch.manual_seed(args.seed + ADJUST_SEED)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Create combinations of column indices to unmask
    n_action_dims = env.action_space.shape[0]
    n_interactions = min(args.max_nway_interact, n_action_dims)

    interaction_col_idxs = []
    interaction_level = 1
    while interaction_level <= n_interactions:
        interaction_col_idxs += [
            x for x in combinations(range(n_action_dims), interaction_level)
        ]
        interaction_level += 1

    # Generate experimental data
    obs, _ = env.reset(seed=(args.seed + ADJUST_SEED))
    episode_reward = 0
    episode_length = 0
    saved_obs = 0
    for idx, unmask_cols in enumerate(interaction_col_idxs):
        # Implement hard cap on number of saved observations
        if saved_obs > args.prepopulate_buffer_hard_cap:
            break
        print(f"CausalEx iteration: {idx+1} / {len(interaction_col_idxs)} ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        mask_array = np.ones((n_action_dims,), dtype=bool)
        mask_array[[col_idx for col_idx in unmask_cols]] = False
        for traj_idx in range(args.max_traj_per_interact):
            # Implement hard cap on number of saved observations
            if saved_obs > args.prepopulate_buffer_hard_cap:
                break
            # Iterate through entire trajectory
            terminations = truncations = False
            while not (terminations or truncations):
                actions = env.action_space.sample()
                actions[mask_array] = 0.
                next_obs, rewards, terminations, truncations, infos = env.step(actions) #TODO: check warning
                
                # Add data to replay buffer
                real_next_obs = next_obs.copy() if not (terminations or truncations) else obs
                rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
                saved_obs += 1
                episode_reward += rewards
                episode_length += 1
                # if "episode" in infos:
                #     print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                #     writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                #     writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

                # Reset environment on termination or truncation
                if terminations or truncations:
                    obs, _ = env.reset() #TODO: consider setting seed here
                    # print(
                    #     f"# interactions: {n_action_dims - sum(mask_array)}; " +
                    #     f"trajectory idx: {traj_idx}; " +
                    #     f"episode reward, length: ({episode_reward}, {episode_length})"
                    # )
                    episode_reward = 0
                    episode_length = 0
                else:
                    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                    obs = next_obs
    return rb


def prepopulate_buffer_random(env, rb, args) -> ReplayBuffer:
    """Prepopulated replay buffer with randomly sampled actions.
    
    Arguments
        env: instantiated gymnasium environment
        rb: instantiated stable_baselines3 replay buffer
        args: configuration arguments dataclass
    
    Returns
        stable_baselines3 replay buffer with trajectory data
    """
    # set up random seeds and device
    random.seed(args.seed + ADJUST_SEED)
    np.random.seed(args.seed + ADJUST_SEED)
    torch.manual_seed(args.seed + ADJUST_SEED)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Calculate number of random trajectories to generate
    # For fair comparison, set equal to the number of trajectories generated for Causal Explorer:
    #   total_unique_interactions = sum({n choose r}) 
    #       for n=action_space_dim, r={1,...,min(action_space_dim, max_nway_interact)}; 
    #   total number of trajectories = total_unique_interactions * max_traj_per_interact
    n_action_dims = env.action_space.shape[0]
    n_interactions = min(args.max_nway_interact, n_action_dims)
    n_unique_interact = sum([comb(n_interactions, r) for r in range(1, n_interactions + 1)])
    n_total_traj = n_unique_interact * args.max_traj_per_interact

    # Generate random data
    obs, _ = env.reset(seed=(args.seed + ADJUST_SEED))
    episode_reward = 0
    episode_length = 0
    saved_obs = 0
    for traj_idx in range(n_total_traj):
        print(f"Random explorer: iter {traj_idx+1} / {n_total_traj}")
        # Implement hard cap on number of saved observations
        if saved_obs > args.prepopulate_buffer_hard_cap:
            break
        # Iterate through entire trajectory
        terminations = truncations = False
        while not (terminations or truncations):
            actions = env.action_space.sample()
            next_obs, rewards, terminations, truncations, infos = env.step(actions) #TODO: check warning
            
            # Add data to replay buffer
            real_next_obs = next_obs.copy() if not (terminations or truncations) else obs
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
            saved_obs += 1
            episode_reward += rewards
            episode_length += 1
            # if "episode" in infos:
            #     print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            #     writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            #     writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

            # Reset environment on termination or truncation
            if terminations or truncations:
                obs, _ = env.reset() #TODO: consider setting seed here
                # print(
                #     f"trajectory idx: {traj_idx}; " +
                #     f"episode reward, length: ({episode_reward}, {episode_length})"
                # )
                episode_reward = 0
                episode_length = 0
            else:
                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs
    return rb