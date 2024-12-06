
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_epoch_losses(run_args):
    """Visualize reward prediction epoch losses averaged across experiments."""
    for env_id in run_args.env_ids:
        fig, axes = plt.subplots(1, len(run_args.test_buffer_sizes), figsize=(16,6))
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        for j, buffer_size in enumerate(run_args.test_buffer_sizes):
            for cx_mode in run_args.cx_modes:
                # Get all experiment folders
                file_pattern = f"epoch_losses_env_{run_args.env_id}_mode_{run_args.cx_mode}_buffer_{run_args.buffer_size}_*"
                files = [f for f in glob.glob(file_pattern) if os.path.isfile(f)]
                # Load epoch losses
                epoch_losses_all = []
                for filename in files:
                    with open(os.path.join(run_args.loss_data_dir, filename), "r") as io:
                        epoch_losses_all.append(json.load(io))
                # Compute average epoch losses across experiments
                loss_data = np.array(epoch_losses_all).T
                avg_loss_data = np.mean(loss_data, axis=1)
                # Plot average epoch losses
                axes[j].plot(avg_loss_data, label=cx_mode)
                axes[j].set_xlabel("Epoch")
                axes[j].set_ylabel("MSE Loss")
                axes[j].set_title(f"Buffer size: {buffer_size:,}")
        axes[-1].legend()
        fig.suptitle(f"Reward Prediction Losses ({env_id})", y=1.05)
        savefig_path = os.path.join(
            run_args.loss_data_dir, f"predict_reward_losses_env_{env_id}.png"
        )
        fig.savefig(savefig_path, bbox_inches="tight")


if __name__ == "__main__":
    from CausalEx.config import RunArgs
    run_args = RunArgs()
    run_args.test_buffer_sizes = [5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    run_args.loss_data_dir = os.path.join("runs", "predict_rewards")
    visualize_epoch_losses(run_args)
