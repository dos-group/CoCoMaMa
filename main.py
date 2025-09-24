import multiprocessing
import pickle
import psutil
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm

from algorithms import CCMAB, CoCoMaMa, ACCUCB, Benchmark, Random, Neural_CoCoMaMa, NeuralMAB
from Hypercube import Hypercube
from Hyperrectangle import Hyperrectangle
from sprout_problem_model import SPROUTProblemModel
from sprout_synthetic_agents_problem_model import SPROUTSyntheticAgentsProblemModel
from plotting import (
    plot_selected_agents,
    plot_all_average_leaves,
    plot_all_cumulative_regret,
    plot_all_average_reward,
    plot_additional_metrics,
)

"""
This python script is responsible for running ACC-UCB, CC-MAB, Random, and benchmark on different problem models.
The script supports both the original SPROUT dataset and the synthetic agents dataset.
"""

# Problem model configuration
PROBLEM_MODEL_TYPE = "sprout"  # Options: "sprout" or "synthetic"
use_saved_data = False  # when True, the script uses saved data instead of generating new data
use_generated_workers_in_paper = False  # Only relevant for SPROUT dataset
only_redo_plots = False  # when True, only redo plots using existing parallel_results files
sns.set(style='whitegrid')

# Results directory
RESULTS_DIR = "results/test_sprout_384_dim"

# Embedding configuration
embedding_config = {
    'model_name': 'all-MiniLM-L6-v2',
    'dimensions': 384,
    'suffix': '_all-MiniLM-L6-v2_384-dim'
}

# Simulation parameters
num_times_to_run = 10
num_rounds = 300
num_std_to_show = 5
budgets = [1,2,3,4]
line_style_dict = {1: '--', 2: '-', 3: '-.',4: ':'}

# Algorithm colors for distinct visualization
algorithm_colors = {
    'HD-ACC-UCB': 'red',
    'CoCoMaMa': 'green', 
    'CC-MAB': 'blue',
    'Neural-CoCoMaMa (ours)': 'cyan',
    'Neural-MAB': 'orange',
    'Random': 'purple',
    'Oracle': 'black'
}

# Algorithm parameters
v1 = np.sqrt(5)
v2 = 1
rho = 0.9
N = 2

# Initialize root context for the algorithms
root_context = Hyperrectangle(np.ones(384*2)*2, np.zeros(384*2))  # For ACC-UCB
root_context_2 = Hypercube(2, np.zeros(384*2))  # For CC-MAB

# Threading configuration
num_threads_to_use = 10  # number of threads to run the simulation on. When set to -1, will run on all available threads

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def run_one_try(problem_model, num_run, budget):
    print(f"Problem model size: {problem_model.get_size()}")
    random_algo = Random(problem_model, budget)
    bench_algo = Benchmark(problem_model, budget)
    omv_ucb_algo = CoCoMaMa(problem_model, v1, v2, N, rho, budget, root_context)
    cc_ucb_algo = ACCUCB(problem_model, v1, v2, N, rho, budget, root_context)
    cc_mab_algo = CCMAB(problem_model, budget, root_context_2.get_dimension())
    neural_cocoma_algo = Neural_CoCoMaMa(problem_model, v1, v2, N, rho, budget, root_context)
    neural_mab_algo = NeuralMAB(problem_model, budget)

    print("Running Neural MAB...")
    neural_mab_reward, neural_mab_regret, neural_mab_played_arms_arr = neural_mab_algo.run_algorithm()
    print("Running Neural CoCoMaMa...")
    neural_cocoma_reward, neural_cocoma_regret, neural_cocoma_played_arms_arr, neural_cocoma_leaves_count_arr, neural_cocoma_metrics = neural_cocoma_algo.run_algorithm()
    print("Running CoCoMaMa...")
    omv_reward, omv_regret, omv_played_arms_arr, omv_leaves_count_arr, omv_metrics = omv_ucb_algo.run_algorithm()
    print("Running ACC-UCB...")
    ucb_reward, ucb_regret, ucb_played_arms_arr, ucb_leaves_count_arr  = cc_ucb_algo.run_algorithm()
    print("Running CC-MAB...")
    cc_mab_reward, cc_mab_regret = cc_mab_algo.run_algorithm()
    print("Running oracle router...")
    bench_reward, bench_regret, bench_played_arms_arr, uniquely_best_arms_arr = bench_algo.run_algorithm()
    print("Running random router...")
    random_reward, random_regret = random_algo.run_algorithm()
    print("Run done: " + str(num_run))

    return {
        'bench_reward': bench_reward,
        'random_reward': random_reward,
        'random_regret': random_regret,
        'ucb_reward': ucb_reward,
        'cocomama_reward': omv_reward,
        'cc_mab_reward': cc_mab_reward,
        'neural_cocoma_reward': neural_cocoma_reward,
        'neural_mab_reward': neural_mab_reward,
        'ucb_regret': ucb_regret,
        'cocomama_regret': omv_regret,
        'cc_mab_regret': cc_mab_regret,
        'neural_cocoma_regret': neural_cocoma_regret,
        'neural_mab_regret': neural_mab_regret,
        'ucb_played_arms_arr': ucb_played_arms_arr,
        'bench_played_arms_arr': bench_played_arms_arr,
        'uniquely_best_arms_arr': uniquely_best_arms_arr,
        'cocomama_played_arms_arr': omv_played_arms_arr,
        'neural_cocoma_played_arms_arr': neural_cocoma_played_arms_arr,
        'neural_mab_played_arms_arr': neural_mab_played_arms_arr,
        'ucb_leaves_count_arr': ucb_leaves_count_arr,
        'cocomama_leaves_count_arr': omv_leaves_count_arr,
        'neural_cocoma_leaves_count_arr': neural_cocoma_leaves_count_arr,
        # New metrics
        # 'ucb_metrics': ucb_metrics,
        'cocomama_metrics': omv_metrics,
        'neural_cocoma_metrics': neural_cocoma_metrics,
        # 'neural_metrics': neural_metrics
    }

def get_problem_model():
    """Create and return the appropriate problem model based on configuration."""
    if PROBLEM_MODEL_TYPE == "sprout":
        return SPROUTProblemModel(num_rounds, max(budgets), use_saved_data, embedding_config)
    elif PROBLEM_MODEL_TYPE == "synthetic":
        return SPROUTSyntheticAgentsProblemModel(
            num_rounds, 
            max(budgets), 
            use_saved_data, 
            embedding_config,
            specialized_agent_interval=200,
            expert_start_time=2000,
            expert_strength_breakpoint=6000,
            num_expert_dimensions=5,
            num_task_fit_dimensions=1
        )
    else:
        raise ValueError(f"Unknown problem model type: {PROBLEM_MODEL_TYPE}")

def should_plot_selected_agents():
    """Determine if we should plot selected agents based on the problem model type."""
    return PROBLEM_MODEL_TYPE == "sprout"  # Only plot for SPROUT dataset as it has fixed agents

def get_agent_names():
    """Get the list of agent names based on the problem model type."""
    if PROBLEM_MODEL_TYPE == "sprout":
        return ["claude-3-5-sonnet-v1",
                "titan-text-premier-v1",
                "gpt-4o",
                "gpt-4o-mini",
                "granite-3-2b",
                "granite-3-8b",
                "llama-3-1-70b",
                "llama-3-1-8b",
                "llama-3-2-1b",
                "llama-3-2-3b",
                "llama-3-3-70b",
                "llama-3-405b",
                "mixtral-8x7b-v01"]
    else:
        return None  # Synthetic dataset has dynamic agents

def run_simulation(problem_model, budget):
    """Run the simulation for a given budget."""
    print(f"Doing budget {budget}...")
    print(f"Memory usage before parallel processing: {get_memory_usage():.2f} MB")
    
    if num_threads_to_use == -1:
        num_threads = int(multiprocessing.cpu_count()-1)
    else:
        num_threads = num_threads_to_use
    print(f"Running on {num_threads} threads")
    
    parallel_results = Parallel(n_jobs=num_threads)(
        delayed(run_one_try)(problem_model, i, budget) for i in tqdm(range(num_times_to_run)))
    
    print(f"Memory usage after parallel processing: {get_memory_usage():.2f} MB")
    
    with open(f'{RESULTS_DIR}/parallel_results_budget_{budget}', 'wb') as output:
        pickle.dump(parallel_results, output, pickle.HIGHEST_PROTOCOL)
    
    return parallel_results

def load_and_process_results():
    """Load and process results for all budgets, returning a dictionary of processed data."""
    all_processed_data = {}
    for budget in budgets:
        with open(f'{RESULTS_DIR}/parallel_results_budget_{budget}', 'rb') as input_file:
            parallel_results = pickle.load(input_file)
        
        num_runs = len(parallel_results)
        ucb_reward_runs_arr = np.zeros((num_runs, num_rounds))
        omv_reward_runs_arr = np.zeros((num_runs, num_rounds))
        cc_mab_reward_runs_arr = np.zeros((num_runs, num_rounds))
        neural_cocoma_reward_runs_arr = np.zeros((num_runs, num_rounds))
        neural_mab_reward_runs_arr = np.zeros((num_runs, num_rounds))
        random_reward_runs_arr = np.zeros((num_runs, num_rounds))
        bench_reward_runs_arr = np.zeros((num_runs, num_rounds))
        
        ucb_regret_runs_arr = np.zeros((num_runs, num_rounds))
        omv_regret_runs_arr = np.zeros((num_runs, num_rounds))
        cc_mab_regret_runs_arr = np.zeros((num_runs, num_rounds))
        neural_cocoma_regret_runs_arr = np.zeros((num_runs, num_rounds))
        neural_mab_regret_runs_arr = np.zeros((num_runs, num_rounds))
        random_regret_runs_arr = np.zeros((num_runs, num_rounds))
        
        ucb_leaves_runs_arr = np.zeros((num_runs, num_rounds))
        omv_leaves_runs_arr = np.zeros((num_runs, num_rounds))
        neural_cocoma_leaves_runs_arr = np.zeros((num_runs, num_rounds))
        
        for i, entry in enumerate(parallel_results):
            # Process rewards
            ucb_reward_runs_arr[i] = pd.Series(entry['ucb_reward']).expanding().mean().values
            omv_reward_runs_arr[i] = pd.Series(entry['cocomama_reward']).expanding().mean().values
            cc_mab_reward_runs_arr[i] = pd.Series(entry['cc_mab_reward']).expanding().mean().values
            neural_cocoma_reward_runs_arr[i] = pd.Series(entry['neural_cocoma_reward']).expanding().mean().values
            neural_mab_reward_runs_arr[i] = pd.Series(entry['neural_mab_reward']).expanding().mean().values
            random_reward_runs_arr[i] = pd.Series(entry['random_reward']).expanding().mean().values
            bench_reward_runs_arr[i] = pd.Series(entry['bench_reward']).expanding().mean().values
            
            # Process regrets
            ucb_regret_runs_arr[i] = np.cumsum(entry['ucb_regret'])
            omv_regret_runs_arr[i] = np.cumsum(entry['cocomama_regret'])
            cc_mab_regret_runs_arr[i] = np.cumsum(entry['cc_mab_regret'])
            neural_cocoma_regret_runs_arr[i] = np.cumsum(entry['neural_cocoma_regret'])
            neural_mab_regret_runs_arr[i] = np.cumsum(entry['neural_mab_regret'])
            random_regret_runs_arr[i] = np.cumsum(entry['random_regret'])
            
            # Process leaves
            ucb_leaves_runs_arr[i] = pd.Series(entry['ucb_leaves_count_arr']).expanding().mean().values
            omv_leaves_runs_arr[i] = pd.Series(entry['cocomama_leaves_count_arr']).expanding().mean().values
            neural_cocoma_leaves_runs_arr[i] = pd.Series(entry['neural_cocoma_leaves_count_arr']).expanding().mean().values
        
        all_processed_data[budget] = {
            'ucb_avg_reward': np.mean(ucb_reward_runs_arr, axis=0),
            'cocomama_avg_reward': np.mean(omv_reward_runs_arr, axis=0),
            'cc_mab_avg_reward': np.mean(cc_mab_reward_runs_arr, axis=0),
            'neural_cocoma_avg_reward': np.mean(neural_cocoma_reward_runs_arr, axis=0),
            'neural_mab_avg_reward': np.mean(neural_mab_reward_runs_arr, axis=0),
            'random_avg_reward': np.mean(random_reward_runs_arr, axis=0),
            'bench_avg_reward': np.mean(bench_reward_runs_arr, axis=0),
            
            'ucb_std_reward': np.std(ucb_reward_runs_arr, axis=0),
            'cocomama_std_reward': np.std(omv_reward_runs_arr, axis=0),
            'cc_mab_std_reward': np.std(cc_mab_reward_runs_arr, axis=0),
            'neural_cocoma_std_reward': np.std(neural_cocoma_reward_runs_arr, axis=0),
            'neural_mab_std_reward': np.std(neural_mab_reward_runs_arr, axis=0),
            'random_std_reward': np.std(random_reward_runs_arr, axis=0),
            'bench_std_reward': np.std(bench_reward_runs_arr, axis=0),
            
            'ucb_avg_regret': np.mean(ucb_regret_runs_arr, axis=0),
            'cocomama_avg_regret': np.mean(omv_regret_runs_arr, axis=0),
            'cc_mab_avg_regret': np.mean(cc_mab_regret_runs_arr, axis=0),
            'neural_cocoma_avg_regret': np.mean(neural_cocoma_regret_runs_arr, axis=0),
            'neural_mab_avg_regret': np.mean(neural_mab_regret_runs_arr, axis=0),
            'random_avg_regret': np.mean(random_regret_runs_arr, axis=0),
            
            'ucb_std_regret': np.std(ucb_regret_runs_arr, axis=0),
            'cocomama_std_regret': np.std(omv_regret_runs_arr, axis=0),
            'cc_mab_std_regret': np.std(cc_mab_regret_runs_arr, axis=0),
            'neural_cocoma_std_regret': np.std(neural_cocoma_regret_runs_arr, axis=0),
            'neural_mab_std_regret': np.std(neural_mab_regret_runs_arr, axis=0),
            'random_std_regret': np.std(random_regret_runs_arr, axis=0),
            
            'ucb_avg_leaves': np.mean(ucb_leaves_runs_arr, axis=0),
            'cocomama_avg_leaves': np.mean(omv_leaves_runs_arr, axis=0),
            'neural_cocoma_avg_leaves': np.mean(neural_cocoma_leaves_runs_arr, axis=0),
            'ucb_std_leaves': np.std(ucb_leaves_runs_arr, axis=0),
            'cocomama_std_leaves': np.std(omv_leaves_runs_arr, axis=0),
            'neural_cocoma_std_leaves': np.std(neural_cocoma_leaves_runs_arr, axis=0),
            'parallel_results': parallel_results # Keep raw results for other plots like boxplots or heatmaps
        }
    return all_processed_data

    

def main():
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    # Run experiment if not only redoing plots
    if not only_redo_plots:
        if not use_saved_data:
            print("Loading problem model...")
            problem_model = get_problem_model()
            print(f"Memory usage after problem model initialization: {get_memory_usage():.2f} MB")
            
            # Check context space bounds
            print("\n=== Context Space Analysis ===")
            sample_arms = problem_model.get_available_arms(1)
            if sample_arms:
                sample_context = sample_arms[0].context
                print(f"Context dimension: {len(sample_context)}")
                print(f"Context min value: {np.min(sample_context):.6f}")
                print(f"Context max value: {np.max(sample_context):.6f}")
                print(f"Context mean value: {np.mean(sample_context):.6f}")
                print(f"Context std value: {np.std(sample_context):.6f}")
                print(f"Context L2 norm: {np.linalg.norm(sample_context):.6f}")
                
                # Check if context is approximately normalized (L2 norm close to 1)
                if abs(np.linalg.norm(sample_context) - 1.0) < 0.1:
                    print("✓ Context appears to be normalized (L2 norm ≈ 1)")
                else:
                    print("⚠ Context is not normalized")
                
                # Check if values are in [-1, 1] range
                if np.all(sample_context >= -1) and np.all(sample_context <= 1):
                    print("✓ Context values are in [-1, 1] range")
                else:
                    print("⚠ Context values are outside [-1, 1] range")
                
                # Check if values are in [0, 1] range (as CC-MAB assumes)
                if np.all(sample_context >= 0) and np.all(sample_context <= 1):
                    print("✓ Context values are in [0, 1] range (matches CC-MAB assumption)")
                else:
                    print("⚠ Context values are NOT in [0, 1] range (CC-MAB may not work correctly)")
                    print("  CC-MAB assumes context space is [0,1] x [0,1] x ... x [0,1]")
                    print("  Consider normalizing contexts to [0,1] range before using CC-MAB")
            print("=== End Context Space Analysis ===\n")
            
            # Run simulations
            for budget in budgets:
                run_simulation(problem_model, budget)
    
    # Load and process results for all budgets
    processed_data = load_and_process_results()

    # Create plots for all budgets in 2x2 grid figures
    plot_all_cumulative_regret(processed_data, budgets, num_rounds, num_std_to_show, algorithm_colors)
    plot_all_average_reward(processed_data, budgets, num_rounds, num_std_to_show, algorithm_colors)
    
    # Plot average leaves for all budgets in a single figure
    plot_all_average_leaves(processed_data, budgets, num_rounds, num_std_to_show, line_style_dict, algorithm_colors)

    # Plot other results that are not part of the 2x2 grids
    plot_additional_metrics(processed_data, budgets, RESULTS_DIR)

    # Plot heatmap conditionally for Sprout problem model and budget 4
    if PROBLEM_MODEL_TYPE == "sprout":
        plot_selected_agents(processed_data, num_rounds, RESULTS_DIR, get_agent_names)
    
    plt.show()

if __name__ == '__main__':
    main()
