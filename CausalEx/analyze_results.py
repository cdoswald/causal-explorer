import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import Args


if __name__ == "__main__":

    # Instantiate arguments
    args = Args()

    # Specify MuJoCo tasks
    env_ids = [
        "Ant-v4",
        "HalfCheetah-v4",
        # "Hopper-v4", #TODO: fix XML file
        "Humanoid-v4",
        # "Walker2d-v4", #TODO: fix XML file
    ]

    # Loop over environments
    for env_id in env_ids:

        # Loop over buffer prepopulation modes
        colors = {"causal":"blue", "random":"green"}
        fig, axes = plt.subplots(1, 2, figsize=(16,6))
        for cx_mode in ["causal", "random"]:

            # Get all experiment folders
            exp_folder_pattern = os.path.join(args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
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
            os.path.join(args.run_dir, f"env_{env_id}_metrics.png"),
            bbox_inches="tight",
        )
