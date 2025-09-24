import matplotlib.pyplot as plt
import numpy as np


def _plot_single_average_reward(ax, budget, ucb_avg, omv_avg, cc_mab_avg, neural_cocoma_avg, neural_mab_avg, random_avg, bench_avg, ucb_std, omv_std, cc_mab_std, neural_cocoma_std, neural_mab_std, random_std, bench_std, num_rounds, num_std_to_show, algorithm_colors):
    """Plot average reward for a specific budget on a given axes."""
    for i in range(len(omv_std)):
        if i == 0 or i % int(num_rounds / num_std_to_show) != 0 and i != len(omv_std) - 1:
            ucb_std[i] = omv_std[i] = cc_mab_std[i] = neural_cocoma_std[i] = neural_mab_std[i] = random_std[i] = bench_std[i] = None

    ax.errorbar(range(1, num_rounds + 1), ucb_avg, yerr=ucb_std,
                 label=f"HD-ACC-UCB", capsize=2, color=algorithm_colors['HD-ACC-UCB'],
                 linestyle='-', linewidth=2)
    ax.errorbar(range(1, num_rounds + 1), omv_avg, yerr=omv_std,
                 label=f"CoCoMaMa (ours)", capsize=2, color=algorithm_colors['CoCoMaMa'],
                 linestyle='-', linewidth=2)
    ax.errorbar(range(1, num_rounds + 1), cc_mab_avg, yerr=cc_mab_std,
                 label=f"CC-MAB", capsize=2, color=algorithm_colors['CC-MAB'],
                 linestyle='-', linewidth=2)
    ax.errorbar(range(1, num_rounds + 1), neural_cocoma_avg, yerr=neural_cocoma_std,
                 label=f"Neural-CoCoMaMa (ours)", capsize=2, color=algorithm_colors['Neural-CoCoMaMa (ours)'],
                 linestyle='-', linewidth=2)
    ax.errorbar(range(1, num_rounds + 1), neural_mab_avg, yerr=neural_mab_std,
                 label=f"Neural-MAB", capsize=2, color=algorithm_colors['Neural-MAB'],
                 linestyle='-', linewidth=2)
    ax.errorbar(range(1, num_rounds + 1), random_avg, yerr=random_std,
                 label=f"Random", capsize=2, color=algorithm_colors['Random'],
                 linestyle='-', linewidth=2)
    ax.errorbar(range(1, num_rounds + 1), bench_avg, yerr=bench_std,
                 label=f"Oracle", capsize=2, color=algorithm_colors['Oracle'],
                 linestyle='-', linewidth=2)

    ax.ticklabel_format(style='plain', axis='y')
    ax.set_xlabel("Arriving task $(t)$")
    ax.set_ylabel("Average task reward up to $t$")
    ax.set_title(f'Average Reward (b = {budget})')


def plot_all_average_reward(processed_data, budgets, num_rounds, num_std_to_show, algorithm_colors, results_dir="."):
    """Create a 2x2 grid figure with average reward plots for all budgets."""
    plt.figure(figsize=(15, 12))
    
    num_plots = len(budgets)
    rows = int(num_plots / 2 + (num_plots % 2))
    cols = 2

    for i, budget in enumerate(budgets):
        ax = plt.subplot(rows, cols, i + 1)
        budget_data = processed_data[budget]
        _plot_single_average_reward(
            ax, budget,
            budget_data['ucb_avg_reward'], budget_data['cocomama_avg_reward'],
            budget_data['cc_mab_avg_reward'],
            budget_data['neural_cocoma_avg_reward'], budget_data['neural_mab_avg_reward'],
            budget_data['random_avg_reward'], budget_data['bench_avg_reward'],
            budget_data['ucb_std_reward'], budget_data['cocomama_std_reward'],
            budget_data['cc_mab_std_reward'],
            budget_data['neural_cocoma_std_reward'], budget_data['neural_mab_std_reward'],
            budget_data['random_std_reward'], budget_data['bench_std_reward'],
            num_rounds, num_std_to_show, algorithm_colors
        )
        if i == 0:
            ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{results_dir}/avg_reward_2x2.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.close()


