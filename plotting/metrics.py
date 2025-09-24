import matplotlib.pyplot as plt


def plot_leaf_node_boxplots(budget, parallel_results, results_dir):
    """Create boxplots for leaf node metrics at 10%, 50%, and 100% completion.

    Parameters
    ----------
    budget : int
        Budget used in the run, for title/filename.
    parallel_results : list[dict]
        Raw per-run entries as produced during simulation.
    results_dir : str
        Directory where the plot PDF will be saved.
    """
    plt.figure(figsize=(15, 7))

    completion_points = [
        ('10%', 'leaves_10', 'leaf_counts_10', 'leaf_variances_10', 'leaf_rewards_10'),
        ('50%', 'leaves_50', 'leaf_counts_50', 'leaf_variances_50', 'leaf_rewards_50'),
        ('100%', 'leaves', 'leaf_counts', 'leaf_variances', 'leaf_rewards')
    ]

    all_variances_data = {point: [] for point, _, _, _, _ in completion_points}
    all_counts_data = {point: [] for point, _, _, _, _ in completion_points}
    all_rewards_data = {point: [] for point, _, _, _, _ in completion_points}

    for entry in parallel_results:
        for point_name, leaves_key, counts_key, variances_key, rewards_key in completion_points:
            leaves = entry['neural_cocoma_metrics'][leaves_key]
            leaf_counts = entry['neural_cocoma_metrics'][counts_key]
            m2_dict = entry['neural_cocoma_metrics'][variances_key]
            avg_rewards = entry['neural_cocoma_metrics'][rewards_key]

            node_variances = []
            node_counts = []
            node_rewards = []

            for node in leaves:
                if node in leaf_counts and leaf_counts[node] > 1:
                    variance = m2_dict[node] / (leaf_counts[node] - 1)
                    node_variances.append(variance)
                    node_counts.append(leaf_counts[node])
                    node_rewards.append(avg_rewards[node])

            all_variances_data[point_name].append(node_variances)
            all_counts_data[point_name].append(node_counts)
            all_rewards_data[point_name].append(node_rewards)

    metrics = [
        ('Leaf Node Variances', all_variances_data, 'Variance'),
        ('Leaf Node Play Counts', all_counts_data, 'Number of Plays'),
        ('Leaf Node Average Rewards', all_rewards_data, 'Average Reward')
    ]

    for i, (title, data_dict, ylabel) in enumerate(metrics, 1):
        ax = plt.subplot(1, 3, i)
        plot_data = []
        labels = []
        for point_name, _, _, _, _ in completion_points:
            combined_data = []
            for run_data in data_dict[point_name]:
                combined_data.extend(run_data)
            plot_data.append(combined_data)
            labels.append(f'{point_name} (n={len(combined_data)})')
        ax.boxplot(plot_data, labels=labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)

        for j, (point_name, leaves_key, _, _, _) in enumerate(completion_points):
            all_leaves = set()
            for entry in parallel_results:
                all_leaves.update(entry['cocomama_metrics'][leaves_key])
            num_leaves = len(all_leaves)
            ax.text(j+1, ax.get_ylim()[0], f'#leaves: {num_leaves}',
                    horizontalalignment='center', verticalalignment='top')

    plt.suptitle(f'Leaf Node Metrics Distribution at Different Completion Points - Budget {budget}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(wspace=0.10)
    plt.savefig(f'{results_dir}/leaf_node_boxplots_budget_{budget}.pdf')
    plt.close()


def plot_additional_metrics(processed_data, budgets, results_dir):
    """Plot additional metrics like leaf node boxplots for all budgets."""
    for budget in budgets:
        budget_data = processed_data[budget]
        current_parallel_results = budget_data['parallel_results']
        plot_leaf_node_boxplots(budget, current_parallel_results, results_dir)


