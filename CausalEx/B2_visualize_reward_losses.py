
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":

    loss_data_dir = os.path.join("runs", "predict_rewards")

    env_ids = [
        "Ant-v4",
        "HalfCheetah-v4",
        # "Hopper-v4", #TODO: fix XML file
        "Humanoid-v4",
        # "Walker2d-v4", #TODO: fix XML file
    ]
    cx_modes = ["causal", "random"]
    test_buffer_sizes = [10_000, 100_000, 500_000, 1_000_000]

    # Plot epoch losses
    for i, env_id in enumerate(env_ids):
        fig, axes = plt.subplots(1, len(test_buffer_sizes), figsize=(20,4))
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        for j, buffer_size in enumerate(test_buffer_sizes):
            for k, cx_mode in enumerate(cx_modes):
                file_name = f"reward_epoch_losses_env_{env_id}_mode_{cx_mode}_buffer_{buffer_size}.json"
                with open(os.path.join(loss_data_dir, file_name), "r") as io:
                    reward_loss_data = json.load(io)
                axes[j].plot(reward_loss_data, label=cx_mode)
                axes[j].set_xlabel("Epoch")
                axes[j].set_ylabel("MSE Loss")
                axes[j].set_title(f"Buffer size: {buffer_size:,}")
        axes[-1].legend()
        fig.suptitle(f"Reward Prediction Losses ({env_id})", y=1.05)
        savefig_path = os.path.join(
            loss_data_dir, f"predict_reward_losses_env_{env_id}.png"
        )
        fig.savefig(savefig_path, bbox_inches="tight")
