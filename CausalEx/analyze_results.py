import glob
from itertools import chain
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
        linewidth = 2
        alpha = 0.2
        fig, axes = plt.subplots(1, 2, figsize=(16,6))

        # Loop over buffer prepopulation modes
        for cx_mode in run_args.cx_modes:

            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]

            # Load metrics
            exp_rewards_all = []
            exp_lengths_all = []
            for exp_dir in exp_dirs:
                exp_rewards_path = os.path.join(exp_dir, "episode_rewards.json")
                if os.path.exists(exp_rewards_path):
                    with open(exp_rewards_path, "r") as io:
                        exp_rewards_all.append(json.load(io))
                exp_lengths_path = os.path.join(exp_dir, "episode_lengths.json")
                if os.path.exists(exp_lengths_path):
                    with open(exp_lengths_path, "r") as io:
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
                linewidth=linewidth,
            )
            sns.lineplot(
                x=range(min_num_episodes),
                y=avg_episode_length,
                ax=axes[1],
                label=f"{cx_mode}",
                color=colors[cx_mode],
                linewidth=linewidth,
            )
            # axes[0].fill_between(
            #     x=range(min_num_episodes),
            #     y1=avg_episode_reward - sd_episode_reward,
            #     y2=avg_episode_reward + sd_episode_reward, 
            #     color=colors[cx_mode],
            #     alpha=alpha,
            # )
            # axes[1].fill_between(
            #     x=range(min_num_episodes),
            #     y1=avg_episode_length - sd_episode_length,
            #     y2=avg_episode_length + sd_episode_length, 
            #     color=colors[cx_mode],
            #     alpha=alpha,
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
        linewidth = 1
        alpha = 0.2
        fig, axes = plt.subplots(4, 1, figsize=(8,8))
        fig.subplots_adjust(hspace=1)

        # Loop over buffer prepopulation modes
        for cx_mode in run_args.cx_modes:

            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]

            # Load loss data
            loss_data_all = []
            for exp_dir in exp_dirs:
                loss_data_path = os.path.join(exp_dir, "loss_data.json")
                if os.path.exists(loss_data_path):
                    with open(loss_data_path, "r") as io:
                        loss_data_all.append(json.load(io))
            
            # Unnest loss data and create matrices of shape (n_steps, m_experiments)
            # Note that actor and alpha are updated x times every x timesteps (so formatting is different)
            critic1_losses = np.array([list(exp['critic1'].values()) for exp in loss_data_all]).T
            critic2_losses = np.array([list(exp['critic2'].values()) for exp in loss_data_all]).T
            actor_losses = np.array([
                list(chain.from_iterable(exp['actor'].values()))
                for exp in loss_data_all
            ]).T
            alpha_losses = np.array([
                list(chain.from_iterable(exp['alpha'].values()))
                for exp in loss_data_all
            ]).T

            # Calculate mean and standard dev of losses across all seeds
            avg_step_critic1_loss = np.mean(critic1_losses, axis=1)
            avg_step_critic2_loss = np.mean(critic2_losses, axis=1)
            avg_step_actor_loss = np.mean(actor_losses, axis=1)
            avg_step_alpha_loss = np.mean(alpha_losses, axis=1)

            sd_step_critic1_loss = np.std(critic1_losses, axis=1)
            sd_step_critic2_loss = np.std(critic2_losses, axis=1)
            sd_step_actor_loss = np.std(actor_losses, axis=1)
            sd_step_alpha_loss = np.std(alpha_losses, axis=1)

            # Plot mean and standard deviation
            sns.lineplot(
                x=range(len(avg_step_critic1_loss)),
                y=avg_step_critic1_loss,
                ax=axes[0],
                label=f"{cx_mode}",
                color=colors[cx_mode],
                linewidth=linewidth,
            )
            sns.lineplot(
                x=range(len(avg_step_critic2_loss)),
                y=avg_step_critic2_loss,
                ax=axes[1],
                label=f"{cx_mode}",
                color=colors[cx_mode],
                linewidth=linewidth,
            )
            sns.lineplot(
                x=range(len(avg_step_actor_loss)),
                y=avg_step_actor_loss,
                ax=axes[2],
                label=f"{cx_mode}",
                color=colors[cx_mode],
                linewidth=linewidth,
            )
            sns.lineplot(
                x=range(len(avg_step_alpha_loss)),
                y=avg_step_alpha_loss,
                ax=axes[3],
                label=f"{cx_mode}",
                color=colors[cx_mode],
                linewidth=linewidth,
            )
            axes[0].fill_between(
                x=range(len(avg_step_critic1_loss)),
                y1=avg_step_critic1_loss - sd_step_critic1_loss,
                y2=avg_step_critic1_loss + sd_step_critic1_loss, 
                color=colors[cx_mode],
                alpha=alpha,
            )
            axes[1].fill_between(
                x=range(len(avg_step_critic2_loss)),
                y1=avg_step_critic2_loss - sd_step_critic2_loss,
                y2=avg_step_critic2_loss + sd_step_critic2_loss, 
                color=colors[cx_mode],
                alpha=alpha,
            )
            axes[2].fill_between(
                x=range(len(avg_step_actor_loss)),
                y1=avg_step_actor_loss - sd_step_actor_loss,
                y2=avg_step_actor_loss + sd_step_actor_loss, 
                color=colors[cx_mode],
                alpha=alpha,
            )
            axes[3].fill_between(
                x=range(len(avg_step_alpha_loss)),
                y1=avg_step_alpha_loss - sd_step_alpha_loss,
                y2=avg_step_alpha_loss + sd_step_alpha_loss, 
                color=colors[cx_mode],
                alpha=alpha,
            )
        for i in range(4):
            # axes[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel("Average Loss \n(Stddev)")
        axes[0].set_title("Critic 1 loss", loc="left")
        axes[1].set_title("Critic 2 loss", loc="left")
        axes[2].set_title("Actor loss", loc="left")
        axes[3].set_title("Alpha loss", loc="left")
        fig.align_ylabels()
        fig.suptitle(f"Environment: {env_id}")
        fig.savefig(
            os.path.join(run_args.run_dir, f"env_{env_id}_losses.png"),
            bbox_inches="tight",
        )
