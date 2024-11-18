from dataclasses import dataclass
import json
import os


# Largely based on CleanRL SAC implementation
@dataclass
class Args:
    # General settings
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Experiment settings
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    cx_mode: str = "causal"
    """method for prepopulating replay buffer; options: {'causal', 'random'}"""
    seed: int = 1
    """seed of the experiment"""

    # Directories
    run_name = "SAC_241118v1"
    """unique name to identify run"""
    run_dir: str = os.path.join("runs", run_name)
    """directory to store all experiment results"""
    exp_dir: str = os.path.join(run_dir, f"env_{env_id}_mode_{cx_mode}_seed_{seed}")
    """subdirectory to store specific experiment results"""

    # RL algorithm-specific arguments
    train_timesteps: int = 100_000
    """total training timesteps of the experiments"""
    eval_timesteps: int = 50_000
    """total evaluation timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    learning_starts: int = 0 # original value = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 64
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 2  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Causal Explorer-specific arguments
    prepopulate_buffer_hard_cap: int = 100_000
    """hard maximum for total number of observations to prepopulate in replay buffer"""
    max_nway_interact: int = 6
    """maximum n-way interactions to test"""
    max_traj_per_interact: int = 5
    """number of trajectories to run per n-way interaction"""


    def save_config(self, path) -> None:
        """Save configuration parameters for reference."""
        with open(path, "w") as file:
            json.dump(self.__dict__, file, indent=4)
    
    def update_exp_dir(self) -> None:
        """Update experiment directory if run directory, env_id, or seed changes."""
        self.exp_dir = os.path.join(
            self.run_dir, f"env_{self.env_id}_mode_{self.cx_mode}_seed_{self.seed}"
        )
    
    def __post_init__(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
