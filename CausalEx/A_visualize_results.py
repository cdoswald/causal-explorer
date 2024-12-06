import glob
import h5py
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define constants
COLORS = {"causal":"blue", "random":"green"}
LINEWIDTH = 1
ALPHA = 0.2


def visualize_episode_rewards(run_args):
    """Visualize average episode rewards across random seeds."""
    for env_id in run_args.env_ids:
        fig, ax = plt.subplots(1, 1, figsize=(16,6))
        for cx_mode in run_args.cx_modes:
            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]
            # Load metrics
            exp_rewards_all = []
            for exp_dir in exp_dirs:
                exp_rewards_path = os.path.join(exp_dir, "metrics.h5")
                if os.path.exists(exp_rewards_path):
                    with h5py.File(exp_rewards_path, "r") as file:
                        exp_rewards_all.append(list(file["episode_rewards"][:]))
            # Clip metrics lists so that all seeds have same number of episodes
            min_num_episodes = min([len(exp) for exp in exp_rewards_all])
            rewards = np.array([exp[:min_num_episodes] for exp in exp_rewards_all]).T
            # Calculate mean and standard dev of metrics across all seeds
            avg_episode_reward = np.mean(rewards, axis=1)
            sd_episode_reward = np.std(rewards, axis=1)
            # Plot mean and standard deviation
            sns.lineplot(
                x=range(min_num_episodes),
                y=avg_episode_reward,
                ax=ax,
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
            )
            ax.fill_between(
                x=range(min_num_episodes),
                y1=avg_episode_reward - sd_episode_reward,
                y2=avg_episode_reward + sd_episode_reward, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
        # Label plot
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Episode Reward")
        ax.set_title(f"Average Episode Reward by Episode ({env_id})")
        fig.savefig(
            os.path.join(run_args.run_dir, f"env_{env_id}_episode_rewards.png"),
            bbox_inches="tight",
        )


def visualize_episode_lengths(run_args):
    """Visualize average episode lengths across random seeds."""
    for env_id in run_args.env_ids:
        fig, ax = plt.subplots(1, 1, figsize=(16,6))
        for cx_mode in run_args.cx_modes:
            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]
            # Load metrics
            exp_lengths_all = []
            for exp_dir in exp_dirs:
                exp_lengths_path = os.path.join(exp_dir, "metrics.h5")
                if os.path.exists(exp_lengths_path):
                    with h5py.File(exp_lengths_path, "r") as file:
                        exp_lengths_all.append(list(file["episode_lengths"][:]))
            # Clip metrics lists so that all seeds have same number of episodes
            min_num_episodes = min([len(exp) for exp in exp_lengths_all])
            lengths = np.array([exp[:min_num_episodes] for exp in exp_lengths_all]).T
            # Calculate mean and standard dev of metrics across all seeds
            avg_episode_length = np.mean(lengths, axis=1)
            sd_episode_length = np.std(lengths, axis=1)
            # Plot mean and standard deviation
            sns.lineplot(
                x=range(min_num_episodes),
                y=avg_episode_length,
                ax=ax,
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
            )
            ax.fill_between(
                x=range(min_num_episodes),
                y1=avg_episode_length - sd_episode_length,
                y2=avg_episode_length + sd_episode_length, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
        # Label plot
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Episode Length")
        ax.set_title(f"Average Episode Length by Episode ({env_id})")
        fig.savefig(
            os.path.join(run_args.run_dir, f"env_{env_id}_episode_lengths.png"),
            bbox_inches="tight",
        )


def visualize_cumulative_rewards(run_args):
    """Visualize cumulative rewards by global timestep across random seeds."""
    for env_id in run_args.env_ids:
        fig, ax = plt.subplots(1, 1, figsize=(16,6))
        for cx_mode in run_args.cx_modes:
            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]
            # Load metrics
            exp_rewards_all = []
            for exp_dir in exp_dirs:
                exp_rewards_path = os.path.join(exp_dir, "metrics.h5")
                if os.path.exists(exp_rewards_path):
                    with h5py.File(exp_rewards_path, "r") as file:
                        exp_rewards_all.append(list(file["cumulative_rewards"][:]))
            rewards = np.array(exp_rewards_all).T
            # Average the cumulative average returns across experiments
            avg_cumul_rewards = np.mean(rewards, axis=1)
            sd_cumul_rewards = np.std(rewards, axis=1)
            # Plot mean and standard deviation
            sns.lineplot(
                x=range(rewards.shape[0]),
                y=avg_cumul_rewards,
                ax=ax,
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
            )
            ax.fill_between(
                x=range(rewards.shape[0]),
                y1=avg_cumul_rewards - sd_cumul_rewards,
                y2=avg_cumul_rewards + sd_cumul_rewards, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
        # Label plot
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax.set_xlabel("Global Timestep")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(f"Cumulative Reward by Timestep ({env_id})")
        fig.savefig(
            os.path.join(run_args.run_dir, f"env_{env_id}_cumul_rewards.png"),
            bbox_inches="tight",
        )


def visualize_model_losses(run_args):
    """Visualize losses for actor and critic models."""
    for env_id in run_args.env_ids:
        fig, axes = plt.subplots(4, 1, figsize=(8,8))
        fig.subplots_adjust(hspace=1)
        for cx_mode in run_args.cx_modes:
            # Get all experiment folders
            exp_folder_pattern = os.path.join(run_args.run_dir, f"env_{env_id}_mode_{cx_mode}_*")
            exp_dirs = [f for f in glob.glob(exp_folder_pattern) if os.path.isdir(f)]
            # Load loss data
            loss_data_all = {"critic1":[], "critic2":[], "actor":[], "alpha":[]}
            for exp_dir in exp_dirs:
                loss_data_path = os.path.join(exp_dir, "loss_data.h5")
                if os.path.exists(loss_data_path):
                    with h5py.File(loss_data_path, "r") as file:
                        loss_data_all["critic1"].append(list(file["critic1"][:]))
                        loss_data_all["critic2"].append(list(file["critic2"][:]))
                        loss_data_all["actor"].append(list(file["actor"][:]))
                        loss_data_all["alpha"].append(list(file["alpha"][:]))
            # Create matries of shape (n_steps, m_experiments)
            critic1_losses = np.array(loss_data_all['critic1']).T
            critic2_losses = np.array(loss_data_all['critic2']).T
            actor_losses = np.array(loss_data_all['actor']).T
            alpha_losses = np.array(loss_data_all['alpha']).T
            # Calculate mean of losses across all seeds
            avg_step_critic1_loss = np.mean(critic1_losses, axis=1)
            avg_step_critic2_loss = np.mean(critic2_losses, axis=1)
            avg_step_actor_loss = np.mean(actor_losses, axis=1)
            avg_step_alpha_loss = np.mean(alpha_losses, axis=1)
            # Calculate standard dev of losses across all seeds
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
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
            )
            sns.lineplot(
                x=range(len(avg_step_critic2_loss)),
                y=avg_step_critic2_loss,
                ax=axes[1],
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
            )
            sns.lineplot(
                x=range(len(avg_step_actor_loss)),
                y=avg_step_actor_loss,
                ax=axes[2],
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
            )
            sns.lineplot(
                x=range(len(avg_step_alpha_loss)),
                y=avg_step_alpha_loss,
                ax=axes[3],
                label=f"{cx_mode}",
                color=COLORS[cx_mode],
                linewidth=LINEWIDTH,
            )
            axes[0].fill_between(
                x=range(len(avg_step_critic1_loss)),
                y1=avg_step_critic1_loss - sd_step_critic1_loss,
                y2=avg_step_critic1_loss + sd_step_critic1_loss, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
            axes[1].fill_between(
                x=range(len(avg_step_critic2_loss)),
                y1=avg_step_critic2_loss - sd_step_critic2_loss,
                y2=avg_step_critic2_loss + sd_step_critic2_loss, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
            axes[2].fill_between(
                x=range(len(avg_step_actor_loss)),
                y1=avg_step_actor_loss - sd_step_actor_loss,
                y2=avg_step_actor_loss + sd_step_actor_loss, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
            )
            axes[3].fill_between(
                x=range(len(avg_step_alpha_loss)),
                y1=avg_step_alpha_loss - sd_step_alpha_loss,
                y2=avg_step_alpha_loss + sd_step_alpha_loss, 
                color=COLORS[cx_mode],
                alpha=ALPHA,
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
        fig.suptitle(f"Model Losses ({env_id})")
        fig.savefig(
            os.path.join(run_args.run_dir, f"env_{env_id}_losses.png"),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    from CausalEx.config import RunArgs
    run_args = RunArgs()
    visualize_episode_rewards(run_args)
    visualize_episode_lengths(run_args)
    visualize_cumulative_rewards(run_args)
    visualize_model_losses(run_args)
