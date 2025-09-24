import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter


def plot_selected_agents(processed_data, num_rounds, results_dir, get_agent_names):
    """Plot a heatmap of agent selection rates for a single run at a fixed budget.

    Parameters
    ----------
    processed_data : dict[int, dict]
        Processed results keyed by budget as created by `load_and_process_results()`.
        Must contain 'parallel_results' with raw run entries for each budget.
    num_rounds : int
        Number of rounds in the experiment, used to normalize counts to rates.
    results_dir : str
        Directory where the plot PDF will be saved.
    get_agent_names : Callable[[], list[str] | None]
        Function returning agent names (for SPROUT) or None for synthetic; used as y-axis labels.
    """
    agent_keys = get_agent_names()
    if agent_keys is None:
        return

    sns.set(style='whitegrid')

    budget = 3
    if budget not in processed_data:
        print(f"Warning: Data for budget {budget} not found. Skipping heatmap generation.")
        return

    entry = processed_data[budget]['parallel_results'][0]

    ucb_played_arms = entry['ucb_played_arms_arr']
    bench_played_arms = entry['bench_played_arms_arr']
    uniquely_best_arms = entry['uniquely_best_arms_arr']
    omv_played_arms = entry['cocomama_played_arms_arr']
    neural_cocoma_played_arms = entry['neural_cocoma_played_arms_arr']
    neural_mab_played_arms = entry['neural_mab_played_arms_arr']

    def extract_unique_ids(algo_runs_data):
        ids = []
        for run in algo_runs_data:
            for round in run:
                for arm in round:
                    ids.append(arm.unique_id)
        return ids

    ids_bin1 = extract_unique_ids([ucb_played_arms])
    ids_bin2 = extract_unique_ids([bench_played_arms])
    ids_bin3 = extract_unique_ids([uniquely_best_arms])
    ids_bin4 = extract_unique_ids([omv_played_arms])
    ids_bin6 = extract_unique_ids([neural_cocoma_played_arms])
    ids_bin7 = extract_unique_ids([neural_mab_played_arms])
    count1 = Counter(ids_bin1)
    count2 = Counter(ids_bin2)
    count3 = Counter(ids_bin3)
    count4 = Counter(ids_bin4)
    count6 = Counter(ids_bin6)
    count7 = Counter(ids_bin7)

    all_ids = list(range(len(agent_keys)))

    columns = [
        ('Oracle Metrics', 'among best'),
        ('Oracle Metrics', 'unique best'),
        ('Algorithm', 'HD-ACC-UCB'),
        ('Algorithm', 'CoCoMaMa (ours)'),
        ('Algorithm', 'Neural-CoCoMaMa (ours)'),
        ('Algorithm', 'Neural-MAB'),
    ]
    multi_columns = pd.MultiIndex.from_tuples(columns)
    data = [
        [count2.get(uid, 0) for uid in all_ids],
        [count3.get(uid, 0) for uid in all_ids],
        [count1.get(uid, 0) for uid in all_ids],
        [count4.get(uid, 0) for uid in all_ids],
        [count6.get(uid, 0) for uid in all_ids],
        [count7.get(uid, 0) for uid in all_ids],
    ]
    df = pd.DataFrame(np.array(data).T, columns=multi_columns, index=[agent_keys[i] for i in all_ids])
    percent_df = df.div(num_rounds, axis=0)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(percent_df, annot=True, cmap="YlGnBu", fmt=".3f", cbar_kws={'label': 'Selection rate'})
    plt.title(f'Agent Selection Rates (b = {budget}) - Single Run')
    plt.ylabel('Agent')
    ax.axvline(x=2, color='black', linewidth=2)
    ax.set_xlabel('')
    ax.set_xticklabels([
        'among best', 'unique best', 'HD-ACC-UCB', 'CoCoMaMa (ours)', 'Neural-CoCoMaMa (ours)', 'Neural-MAB'
    ], rotation=30, ha='right')

    ax.figure.subplots_adjust(top=0.90)
    ax2 = ax.twiny()
    ax2.set_xticks([1, 4])
    ax2.set_xticklabels(["Oracle Metrics", "Algorithm"], ha='center', fontsize=13)
    ax2.set_xlim(ax.get_xlim())
    ax2.tick_params(axis='x', length=0)
    ax2.grid(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/selected_agents_K_{budget}_single_run.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.close()


