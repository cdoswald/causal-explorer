    # Analyze results
    ##------------------##
    # Plot episode rewards and lengths
    for metric in ["rewards", "lengths"]:
        fig, axes = plt.subplots(1, 2, figsize=(8,6))
        for cx_mode in ["causal", "random"]:
            # Generate run name
            train_args.exp_name = f"SAC_train_{cx_mode}{exp_suffix}"
            train_args.gen_run_name()
            eval_args.exp_name = f"SAC_eval_{cx_mode}{exp_suffix}"
            eval_args.gen_run_name()
            # Load data
            with open(f"runs/{train_args.run_name}/episode_{metric}.json", "r") as io:
                train_episode_data = json.load(io)
            with open(f"runs/{eval_args.run_name}/episode_{metric}.json", "r") as io:
                eval_episode_data = json.load(io)
            # Plot data
            sns.lineplot(
                x=range(len(train_episode_data)),
                y=train_episode_data,
                ax=axes[0],
                label=cx_mode,
            )
            sns.lineplot(
                x=range(len(eval_episode_data)),
                y=eval_episode_data,
                ax=axes[1],
                label=cx_mode,
            )
        axes[0].set_title("Training")
        axes[1].set_title("Evaluation")
        fig.suptitle(f"Episode {metric}: {env_id.title()}")
        fig.savefig(f"runs/{eval_args.run_name}/episode_{metric}.png")