import matplotlib.pyplot as plt


def plot_all_average_leaves(processed_data, budgets, num_rounds, num_std_to_show, line_style_dict, algorithm_colors, results_dir="."):
    """Create a 2x2 grid figure with average leaves plots for all budgets.

    Parameters
    ----------
    processed_data : dict[int, dict]
        Processed results per budget.
    budgets : list[int]
        Budgets to plot.
    num_rounds : int
        Number of rounds.
    num_std_to_show : int
        Downsampling factor to reduce errorbar clutter.
    line_style_dict : dict[int, str]
        Mapping from budget to line style.
    algorithm_colors : dict[str, str]
        Colors for algorithms.
    """
    plt.figure(figsize=(15, 12))
    
    num_plots = len(budgets)
    rows = int(num_plots / 2 + (num_plots % 2))
    cols = 2

    for i, budget in enumerate(budgets):
        ax = plt.subplot(rows, cols, i + 1)
        budget_data = processed_data[budget]
        ucb_avg = budget_data['ucb_avg_leaves']
        omv_avg = budget_data['cocomama_avg_leaves']
        neural_cocoma_avg = budget_data['neural_cocoma_avg_leaves']
        ucb_std = budget_data['ucb_std_leaves']
        omv_std = budget_data['cocomama_std_leaves']
        neural_cocoma_std = budget_data['neural_cocoma_std_leaves']

        for j in range(len(omv_std)):
            if j == 0 or j % int(num_rounds / num_std_to_show) != 0 and j != len(omv_std) - 1:
                ucb_std[j] = omv_std[j] = neural_cocoma_std[j] = None

        ax.errorbar(range(1, num_rounds + 1), ucb_avg, yerr=ucb_std,
                   label=f"HD-ACC-UCB", capsize=2, color='r',
                   linestyle='-', linewidth=2)
        ax.errorbar(range(1, num_rounds + 1), omv_avg, yerr=omv_std,
                   label=f"CoCoMaMa (ours)", capsize=2, color=algorithm_colors['CoCoMaMa'],
                   linestyle='-', linewidth=2)
        ax.errorbar(range(1, num_rounds + 1), neural_cocoma_avg, yerr=neural_cocoma_std,
                   label=f"Neural-CoCoMaMa (ours)", capsize=2, color='c',
                   linestyle='-', linewidth=2)

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_xlabel("Arriving task $(t)$")
        ax.set_ylabel("Average leaves up to $t$")
        ax.set_title(f'Average Leaves (b = {budget})')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{results_dir}/avg_leaves_all_budgets.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.close()


