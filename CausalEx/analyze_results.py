import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import RunArgs


if __name__ == "__main__":

    # Specify formatting constants
    colors = {"causal":"blue", "random":"green"}

    # Instantiate run arguments (applies to all experiments)
    run_args = RunArgs()

    # Loop over environments
    for env_id in run_args.env_ids:

        ###################################
        ## Average Episode Reward/Length ##
        ###################################
        # Loop over buffer prepopulation modes
        fig, axes = plt.subplots(1, 2, figsize=(16,6))
        for cx_mode in run_args.cx_modes:

            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]

            # Load metrics
            exp_rewards_all = []
            exp_lengths_all = []
            for exp_dir in exp_dirs:
                with open(os.path.join(exp_dir, f"episode_rewards.json"), "r") as io:
                    exp_rewards_all.append(json.load(io))
                with open(os.path.join(exp_dir, "episode_lengths.json"), "r") as io:
                    exp_lengths_all.append(json.load(io))

            # Clip metrics lists so that all seeds have same number of episodes
            min_num_episodes = min([len(exp) for exp in exp_rewards_all])
            rewards = np.array([exp[:min_num_episodes] for exp in exp_rewards_all]).T
            lengths = np.array([exp[:min_num_episodes] for exp in exp_lengths_all]).T

            # Calculate mean and standard dev of metrics across all seeds
            avg_episode_reward = np.mean(rewards, axis=1)
            avg_episode_length = np.mean(lengths, axis=1)
            sd_episode_reward = np.std(rewards, axis=1)
            sd_episode_length = np.std(lengths, axis=1)

            # Plot mean and standard deviation
            sns.lineplot(
                x=range(min_num_episodes),
                y=avg_episode_reward,
                ax=axes[0],
                label=f"{cx_mode}",
                color=colors[cx_mode],
                linewidth=2,
            )
            sns.lineplot(
                x=range(min_num_episodes),
                y=avg_episode_length,
                ax=axes[1],
                label=f"{cx_mode}",
                color=colors[cx_mode],
                linewidth=2,
            )
            # axes[0].fill_between(
            #     x=range(min_num_episodes),
            #     y1=avg_episode_reward - sd_episode_reward,
            #     y2=avg_episode_reward + sd_episode_reward, 
            #     color=colors[cx_mode],
            #     alpha=0.2
            # )
            # axes[1].fill_between(
            #     x=range(min_num_episodes),
            #     y1=avg_episode_length - sd_episode_length,
            #     y2=avg_episode_length + sd_episode_length, 
            #     color=colors[cx_mode],
            #     alpha=0.2
            # )
        for i in range(2):
            axes[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
            axes[i].set_xlabel("Episode")
        axes[0].set_title("Average episode reward")
        axes[1].set_title("Average episode length")
        fig.suptitle(f"Environment: {env_id}")
        fig.savefig(
            os.path.join(run_args.run_dir, f"env_{env_id}_metrics.png"),
            bbox_inches="tight",
        )

        ###################################
        ## Actor and Critic Model Losses ##
        ###################################
        # Loop over buffer prepopulation modes
        fig, axes = plt.subplots(1, 4, figsize=(16,6))
        for cx_mode in run_args.cx_modes:

            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]

            # Load loss data
            loss_data_all = []
            for exp_dir in exp_dirs:
                with open(os.path.join(exp_dir, f"loss_data.json"), "r") as io:
                    loss_data_all.append(json.load(io))
            
            # Unnest loss data and create matrices of shape (n_steps, m_experiments)
            ## TODO: truncate for same size?
            critic1_losses = np.array([v for exp in loss_data_all for v in exp["critic1"].values()]).T
            critic2_losses = np.array([v for exp in loss_data_all for v in exp["critic2"].values()]).T

            # Structure of loss data dict
        #     {
        #         critic1: {
        #             step1: loss,
        #             ...
        #         },
        #         critic2: {
        #             step1: loss,
        #             ...
        #         },
        #         actor: {
        #             step64: [loss1, loss2, ...],
        #             step128: [loss1, loss2, ...],
        #             ...
        #         },
        #         alpha: {
        #             step64: [loss1, loss2, ...],
        #             step128: [loss1, loss2, ...],
        #             ...
        #         },
        #     }
        # ]



            # # Clip metrics lists so that all seeds have same number of episodes
            # min_num_episodes = min([len(exp) for exp in exp_rewards_all])
            # rewards = np.array([exp[:min_num_episodes] for exp in exp_rewards_all]).T
            # lengths = np.array([exp[:min_num_episodes] for exp in exp_lengths_all]).T

            # # Calculate mean and standard dev of metrics across all seeds
            # avg_episode_reward = np.mean(rewards, axis=1)
            # avg_episode_length = np.mean(lengths, axis=1)
            # sd_episode_reward = np.std(rewards, axis=1)
            # sd_episode_length = np.std(lengths, axis=1)

            