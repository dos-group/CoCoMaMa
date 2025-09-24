"""
Main script for running streaming contextual bandit experiments.
This script supports both traditional problem models and streaming Arrow datasets.
"""

import multiprocessing
import pickle
import psutil
import os
import argparse
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm

# Import traditional algorithms
from algorithms import CCMAB, CoCoMaMa, ACCUCB, Benchmark, Random, Neural_CoCoMaMa, NeuralMAB

# Import streaming algorithms
from algorithms.streaming_base import StreamingRandom, StreamingBenchmark
from algorithms.streaming_cocoma import StreamingCoCoMaMa
from algorithms.streaming_neural_cocomama import StreamingNeuralCoCoMaMa
from algorithms.streaming_hd_acc_ucb import StreamingHDACCUCB
from algorithms.streaming_cc_mab import StreamingCCMAB
from algorithms.streaming_neural_mab import StreamingNeuralMAB

# Import problem models
from Hypercube import Hypercube
from Hyperrectangle import Hyperrectangle
from sprout_problem_model import SPROUTProblemModel
from sprout_synthetic_agents_problem_model import SPROUTSyntheticAgentsProblemModel
from streaming_dataset import StreamingProblemModel

# Import plotting functions
from plotting import (
    plot_selected_agents,
    plot_all_average_leaves,
    plot_all_cumulative_regret,
    plot_all_average_reward,
    plot_additional_metrics,
)

"""
This python script is responsible for running contextual bandit algorithms on streaming datasets.
The script supports both traditional problem models and streaming Arrow datasets.
"""

def load_config_file(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config file: {e}")

def merge_configs(config_file, cli_args):
    """Merge config file with CLI arguments. CLI arguments take precedence."""
    # Start with config file
    merged_config = config_file.copy()
    
    # Override with CLI arguments (only if they are not None)
    for key, value in cli_args.items():
        if value is not None:
            merged_config[key] = value
    
    return merged_config

def save_config_to_file(config, use_streaming_algorithms, filename=None):
    """Save configuration to a YAML file for reproducibility."""
    import datetime
    
    # Create configs directory if it doesn't exist
    os.makedirs("configs", exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        problem_type = config.get("PROBLEM_MODEL_TYPE", "unknown")
        rounds = config.get("num_rounds", 100)
        filename = f"configs/config_{problem_type}_{rounds}rounds_{timestamp}.yaml"
    
    # Prepare config for saving (remove non-serializable items)
    save_config = config.copy()
    
    # Add streaming algorithms flag to config
    save_config["use_streaming_algorithms"] = use_streaming_algorithms
    
    # Remove any non-serializable items
    if "create_dataset_if_missing" in save_config:
        del save_config["create_dataset_if_missing"]
    
    # Save to file
    try:
        with open(filename, 'w') as f:
            yaml.dump(save_config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Configuration saved to: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return None

def get_default_config():
    """Get default configuration as fallback."""
    return {
        "PROBLEM_MODEL_TYPE": "streaming_sprout",
        "use_saved_data": False,
        "use_generated_workers_in_paper": False,
        "only_redo_plots": False,
        "RESULTS_DIR": "results/test_2_streaming_384_dim",
        "embedding_config": {
            'model_name': 'all-MiniLM-L6-v2',
            'dimensions': 384,
            'suffix': '_all-MiniLM-L6-v2_384-dim'
        },
        "num_times_to_run": 3,
        "num_rounds": 100,
        "num_std_to_show": 5,
        "budgets": [1, 2, 3, 4],
        "v1": np.sqrt(5),
        "v2": 1,
        "rho": 0.9,
        "N": 2,
        "num_threads_to_use": 10
    }

def get_user_input(prompt, input_type=str, default=None, choices=None, allow_empty=False):
    """Get user input with validation and default values."""
    while True:
        if choices:
            choice_str = f" ({'/'.join(choices)})" if choices else ""
            default_str = f" [default: {default}]" if default is not None else ""
            full_prompt = f"{prompt}{choice_str}{default_str}: "
        else:
            default_str = f" [default: {default}]" if default is not None else ""
            full_prompt = f"{prompt}{default_str}: "
        
        try:
            user_input = input(full_prompt).strip()
            
            # Handle empty input
            if not user_input:
                if default is not None:
                    return default
                elif allow_empty:
                    return None
                else:
                    print("This field is required. Please enter a value.")
                    continue
            
            # Convert input type
            if input_type == int:
                result = int(user_input)
            elif input_type == float:
                result = float(user_input)
            elif input_type == bool:
                result = user_input.lower() in ['true', 'yes', 'y', '1', 'on']
            elif input_type == list:
                # For lists, split by comma and strip whitespace
                result = [item.strip() for item in user_input.split(',')]
            else:
                result = user_input
            
            # Validate choices
            if choices and result not in choices:
                print(f"Invalid choice. Please choose from: {', '.join(choices)}")
                continue
            
            return result
            
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")
            continue

def interactive_configuration():
    """Interactive configuration setup."""
    print("\n" + "="*60)
    print("🎯 INTERACTIVE CONFIGURATION SETUP")
    print("="*60)
    print("Configure your streaming contextual bandit experiment.")
    print("Press Enter to use default values shown in brackets.\n")
    
    config = {}
    
    # Problem model type
    print("📊 PROBLEM MODEL CONFIGURATION")
    print("-" * 30)
    config["PROBLEM_MODEL_TYPE"] = get_user_input(
        "Problem model type", 
        choices=["sprout", "streaming_sprout", "streaming_synthetic"], 
        default="streaming_sprout"
    )
    
    # Experiment parameters
    print("\n🔬 EXPERIMENT PARAMETERS")
    print("-" * 25)
    config["num_rounds"] = get_user_input(
        "Number of rounds", 
        input_type=int, 
        default=300
    )
    
    config["num_times_to_run"] = get_user_input(
        "Number of runs (for statistical significance)", 
        input_type=int, 
        default=5
    )
    
    # Budgets
    budget_input = get_user_input(
        "Budgets to test (comma-separated)", 
        input_type=list, 
        default=["1", "2", "3", "4"]
    )
    config["budgets"] = [int(b) for b in budget_input]
    
    
    # Algorithm parameters
    print("\n⚙️ ALGORITHM PARAMETERS")
    print("-" * 25)
    config["v1"] = get_user_input(
        "V1 parameter", 
        input_type=float, 
        default=2.23606797749979
    )
    
    config["v2"] = get_user_input(
        "V2 parameter", 
        input_type=float, 
        default=1.0
    )
    
    config["rho"] = get_user_input(
        "Rho parameter", 
        input_type=float, 
        default=0.9
    )
    
    config["N"] = get_user_input(
        "N parameter", 
        input_type=int, 
        default=2
    )
    
    # Embedding configuration
    print("\n🧠 EMBEDDING CONFIGURATION")
    print("-" * 25)
    embedding_model = get_user_input(
        "Embedding model name", 
        default="all-MiniLM-L6-v2"
    )
    
    embedding_dimensions = get_user_input(
        "Embedding dimensions", 
        input_type=int, 
        default=384
    )
    
    config["embedding_config"] = {
        'model_name': embedding_model,
        'dimensions': embedding_dimensions,
        'suffix': f"_{embedding_model}_{embedding_dimensions}-dim"
    }
    
    # Performance configuration
    print("\n🚀 PERFORMANCE CONFIGURATION")
    print("-" * 28)
    config["num_threads_to_use"] = get_user_input(
        "Number of threads (-1 for all available)", 
        input_type=int, 
        default=8
    )
    
    # Output configuration
    print("\n📁 OUTPUT CONFIGURATION")
    print("-" * 22)
    config["RESULTS_DIR"] = get_user_input(
        "Results directory", 
        default="results/interactive_experiment"
    )
    
    # Dataset paths and configuration
    if config["PROBLEM_MODEL_TYPE"] == "streaming_sprout":
        config["streaming_dataset_path"] = get_user_input(
            "SPROUT streaming dataset path", 
            default="datasets/sprout_streaming_300.arrow"
        )
    elif config["PROBLEM_MODEL_TYPE"] == "streaming_synthetic":
        # Synthetic dataset configuration
        print("\n🎭 SYNTHETIC DATASET CONFIGURATION")
        print("-" * 35)
        
        config["synthetic_agents_config"] = {
            "specialized_agent_interval": get_user_input(
                "Specialized agent interval", 
                input_type=int, 
                default=200
            ),
            "expert_start_time": get_user_input(
                "Expert start time", 
                input_type=int, 
                default=2000
            ),
            "expert_strength_breakpoint": get_user_input(
                "Expert strength breakpoint", 
                input_type=int, 
                default=6000
            ),
            "num_expert_dimensions": get_user_input(
                "Number of expert dimensions", 
                input_type=int, 
                default=5
            ),
            "num_task_fit_dimensions": get_user_input(
                "Number of task fit dimensions", 
                input_type=int, 
                default=1
            )
        }
        
        # Ask if user wants to create dataset if it doesn't exist
        create_dataset = get_user_input(
            "Create synthetic dataset if it doesn't exist", 
            input_type=bool, 
            default=True
        )
        config["create_dataset_if_missing"] = create_dataset
        
        if create_dataset:
            config["sprout_streaming_path"] = get_user_input(
                "SPROUT streaming dataset path (for synthetic generation)", 
                default="datasets/sprout_streaming_300.arrow"
            )
            config["synthetic_dataset_path"] = get_user_input(
                "Synthetic dataset output path", 
                default="datasets/synthetic_streaming_300.arrow"
            )
    
    # Flags
    print("\n🎛️ EXPERIMENT FLAGS")
    print("-" * 18)
    config["use_saved_data"] = get_user_input(
        "Use saved data instead of generating new", 
        input_type=bool, 
        default=False
    )
    
    config["use_generated_workers_in_paper"] = get_user_input(
        "Use generated workers from paper", 
        input_type=bool, 
        default=False
    )
    
    config["only_redo_plots"] = get_user_input(
        "Only redo plots using existing results", 
        input_type=bool, 
        default=False
    )
    
    use_streaming_algorithms = get_user_input(
        "Use streaming algorithms", 
        input_type=bool, 
        default=True
    )
    
    config["plot"] = get_user_input(
        "Generate plots after running experiments", 
        input_type=bool, 
        default=True
    )
    
    print("\n" + "="*60)
    print("✅ CONFIGURATION COMPLETE")
    print("="*60)
    
    # Ask if user wants to save configuration
    print("\n💾 SAVE CONFIGURATION")
    print("-" * 20)
    save_config = get_user_input(
        "Save this configuration to a file for reproducibility", 
        input_type=bool, 
        default=True
    )
    
    if save_config:
        custom_filename = get_user_input(
            "Custom filename (leave empty for auto-generated)", 
            allow_empty=True
        )
        
        if custom_filename:
            # Ensure it has .yaml extension
            if not custom_filename.endswith('.yaml') and not custom_filename.endswith('.yml'):
                custom_filename += '.yaml'
            filename = f"configs/{custom_filename}"
        else:
            filename = None
        
        saved_file = save_config_to_file(config, use_streaming_algorithms, filename)
        if saved_file:
            print(f"\n🔄 To reuse this configuration, run:")
            print(f"   python main_streaming.py --config_file {saved_file}")
    
    return config, use_streaming_algorithms

sns.set(style='whitegrid')

# Algorithm colors for distinct visualization
algorithm_colors = {
    'HD-ACC-UCB': 'red',
    'CoCoMaMa': 'green', 
    'CC-MAB': 'blue',
    'Neural-CoCoMaMa (ours)': 'cyan',
    'Neural-MAB': 'orange',
    'Random': 'purple',
    'Oracle': 'black',
    'Streaming-CoCoMaMa': 'brown',
    'Streaming-Neural-MAB': 'pink',
    'Streaming-Random': 'gray'
}

line_style_dict = {1: '--', 2: '-', 3: '-.', 4: ':'}


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_one_try_ablation(num_run, budget, config):
    """Run one trial for ablation study with specific algorithm (no benchmark)."""
    # Create problem model instance
    problem_model_ablation = get_problem_model(config)
    
    print(f"Problem model size: {problem_model_ablation.get_size()}")
    
    # Initialize root context
    root_context = Hyperrectangle(np.ones(config["embedding_config"]["dimensions"]*2)*2, np.zeros(config["embedding_config"]["dimensions"]*2))
    
    # Initialize the ablation algorithm based on config
    ablation_algorithm = config["ablation_algorithm"]
    if ablation_algorithm == "streaming_neural_cocoma":
        streaming_ablation = StreamingNeuralCoCoMaMa(
            problem_model_ablation, config["v1"], config["v2"], config["N"], 
            config["rho"], budget, root_context, config["embedding_config"]["dimensions"]*2, 
            hidden_dim=config.get("hidden_dim", 64)
        )
    elif ablation_algorithm == "streaming_cocoma":
        streaming_ablation = StreamingCoCoMaMa(
            problem_model_ablation, config["v1"], config["v2"], config["N"], 
            config["rho"], budget, root_context, config.get("theta", 4.0)
        )
    elif ablation_algorithm == "streaming_neural_mab":
        streaming_ablation = StreamingNeuralMAB(
            problem_model_ablation, budget, config["embedding_config"]["dimensions"]*2, 
            hidden_dim=config.get("hidden_dim", 64)
        )
    else:
        raise ValueError(f"Unknown ablation algorithm: {ablation_algorithm}")
    
    # Run ablation algorithm
    print(f"Running {ablation_algorithm}...")
    if ablation_algorithm == "streaming_neural_cocoma":
        ablation_reward, ablation_regret, ablation_played_arms, ablation_leaves_count, ablation_metrics = streaming_ablation.run_algorithm()
    elif ablation_algorithm == "streaming_cocoma":
        ablation_reward, ablation_regret, ablation_played_arms, ablation_leaves_count, ablation_metrics = streaming_ablation.run_algorithm()
    elif ablation_algorithm == "streaming_neural_mab":
        ablation_reward, ablation_regret, ablation_played_arms = streaming_ablation.run_algorithm()
        ablation_leaves_count = [0] * len(ablation_reward)
        ablation_metrics = {}
    
    print(f"Run done: {num_run}")
    
    # Return results with ablation algorithm name
    variant_name = config.get("ablation_variant_name", ablation_algorithm)
    results = {
        f'{ablation_algorithm}_reward': ablation_reward,
        f'{ablation_algorithm}_regret': ablation_regret,
        f'{ablation_algorithm}_played_arms_arr': ablation_played_arms,
        f'{ablation_algorithm}_leaves_count_arr': ablation_leaves_count,
        f'{ablation_algorithm}_metrics': ablation_metrics
    }
    
    return results


def run_one_try_streaming(num_run, budget, config):
    """Run one trial with streaming algorithms."""
    # Check if this is an ablation study
    if "ablation_algorithm" in config:
        return run_one_try_ablation(num_run, budget, config)
    
    # Create separate problem model instances for each algorithm to avoid state sharing
    problem_model_random = get_problem_model(config)
    problem_model_benchmark = get_problem_model(config)
    problem_model_cocoma = get_problem_model(config)
    problem_model_neural_mab = get_problem_model(config)
    problem_model_neural_cocoma = get_problem_model(config)
    problem_model_acc_ucb = get_problem_model(config)
    problem_model_cc_mab = get_problem_model(config)
    
    print(f"Problem model size: {problem_model_random.get_size()}")
    
    # Initialize root contexts
    root_context = Hyperrectangle(np.ones(config["embedding_config"]["dimensions"]*2)*2, np.zeros(config["embedding_config"]["dimensions"]*2))
    root_context_2 = Hypercube(2, np.zeros(config["embedding_config"]["dimensions"]*2))
    
    # Initialize streaming algorithms with separate problem models
    streaming_random = StreamingRandom(problem_model_random, budget)
    streaming_benchmark = StreamingBenchmark(problem_model_benchmark, budget)
    streaming_cocoma = StreamingCoCoMaMa(
        problem_model_cocoma, config["v1"], config["v2"], config["N"], 
        config["rho"], budget, root_context, config.get("theta", 4.0)
    )
    streaming_neural_mab = StreamingNeuralMAB(
        problem_model_neural_mab, budget, config["embedding_config"]["dimensions"]*2, hidden_dim=64
    )
    streaming_neural_cocoma = StreamingNeuralCoCoMaMa(
        problem_model_neural_cocoma, config["v1"], config["v2"], config["N"], 
        config["rho"], budget, root_context, config["embedding_config"]["dimensions"]*2, hidden_dim=64
    )
    streaming_hd_acc_ucb = StreamingHDACCUCB(
        problem_model_acc_ucb, config["v1"], config["v2"], config["N"], 
        config["rho"], budget, root_context
    )
    streaming_cc_mab = StreamingCCMAB(
        problem_model_cc_mab, budget, config["embedding_config"]["dimensions"]*2
    )
    
    print("Running Streaming Random...")
    streaming_random_reward, streaming_random_regret, streaming_random_played_arms_arr = streaming_random.run_algorithm()
    
    print("Running Streaming Benchmark...")
    streaming_bench_reward, streaming_bench_regret, streaming_bench_played_arms_arr, streaming_uniquely_best_arms_arr = streaming_benchmark.run_algorithm()
    
    print("Running Streaming CoCoMaMa...")
    streaming_cocoma_reward, streaming_cocoma_regret, streaming_cocoma_played_arms_arr, streaming_cocoma_leaves_count_arr, streaming_cocoma_metrics = streaming_cocoma.run_algorithm()
        
    print("Running Streaming Neural CoCoMaMa...")
    streaming_neural_cocoma_reward, streaming_neural_cocoma_regret, streaming_neural_cocoma_played_arms_arr, streaming_neural_cocoma_leaves_count_arr, streaming_neural_cocoma_metrics = streaming_neural_cocoma.run_algorithm()
    
    print("Running Streaming HD-ACC-UCB...")
    streaming_hd_acc_ucb_reward, streaming_hd_acc_ucb_regret, streaming_hd_acc_ucb_played_arms_arr, streaming_hd_acc_ucb_leaves_count_arr = streaming_hd_acc_ucb.run_algorithm()
    
    print("Running Streaming CC-MAB...")
    streaming_cc_mab_reward, streaming_cc_mab_regret, streaming_cc_mab_played_arms_arr = streaming_cc_mab.run_algorithm()
    
    print("Running Streaming Neural MAB...")
    streaming_neural_mab_reward, streaming_neural_mab_regret, streaming_neural_mab_played_arms_arr = streaming_neural_mab.run_algorithm()
    
    print("Run done: " + str(num_run))
    
    return {
        'streaming_bench_reward': streaming_bench_reward,
        'streaming_random_reward': streaming_random_reward,
        'streaming_random_regret': streaming_random_regret,
        'streaming_cocoma_reward': streaming_cocoma_reward,
        'streaming_cocoma_regret': streaming_cocoma_regret,
        'streaming_neural_cocoma_reward': streaming_neural_cocoma_reward,
        'streaming_neural_cocoma_regret': streaming_neural_cocoma_regret,
        'streaming_hd_acc_ucb_reward': streaming_hd_acc_ucb_reward,
        'streaming_hd_acc_ucb_regret': streaming_hd_acc_ucb_regret,
        'streaming_cc_mab_reward': streaming_cc_mab_reward,
        'streaming_cc_mab_regret': streaming_cc_mab_regret,
        'streaming_neural_mab_reward': streaming_neural_mab_reward,
        'streaming_neural_mab_regret': streaming_neural_mab_regret,
        'streaming_bench_played_arms_arr': streaming_bench_played_arms_arr,
        'streaming_uniquely_best_arms_arr': streaming_uniquely_best_arms_arr,
        'streaming_cocoma_played_arms_arr': streaming_cocoma_played_arms_arr,
        'streaming_neural_cocoma_played_arms_arr': streaming_neural_cocoma_played_arms_arr,
        'streaming_hd_acc_ucb_played_arms_arr': streaming_hd_acc_ucb_played_arms_arr,
        'streaming_cc_mab_played_arms_arr': streaming_cc_mab_played_arms_arr,
        'streaming_neural_mab_played_arms_arr': streaming_neural_mab_played_arms_arr,
        'streaming_cocoma_leaves_count_arr': streaming_cocoma_leaves_count_arr,
        'streaming_neural_cocoma_leaves_count_arr': streaming_neural_cocoma_leaves_count_arr,
        'streaming_hd_acc_ucb_leaves_count_arr': streaming_hd_acc_ucb_leaves_count_arr,
        'streaming_cocoma_metrics': streaming_cocoma_metrics,
        'streaming_neural_cocoma_metrics': streaming_neural_cocoma_metrics
    }


def run_one_try_traditional(problem_model, num_run, budget, config):
    """Run one trial with traditional algorithms."""
    print(f"Problem model size: {problem_model.get_size()}")
    
    # Initialize root contexts
    root_context = Hyperrectangle(np.ones(config["embedding_config"]["dimensions"]*2)*2, np.zeros(config["embedding_config"]["dimensions"]*2))
    root_context_2 = Hypercube(2, np.zeros(config["embedding_config"]["dimensions"]*2))
    
    # Initialize traditional algorithms
    random_algo = Random(problem_model, budget)
    bench_algo = Benchmark(problem_model, budget)
    omv_ucb_algo = CoCoMaMa(problem_model, config["v1"], config["v2"], config["N"], config["rho"], budget, root_context)
    cc_ucb_algo = ACCUCB(problem_model, config["v1"], config["v2"], config["N"], config["rho"], budget, root_context)
    cc_mab_algo = CCMAB(problem_model, budget, root_context_2.get_dimension())
    neural_cocoma_algo = Neural_CoCoMaMa(problem_model, config["v1"], config["v2"], config["N"], config["rho"], budget, root_context)
    neural_mab_algo = NeuralMAB(problem_model, budget)
    
    print("Running Neural MAB...")
    neural_mab_reward, neural_mab_regret, neural_mab_played_arms_arr = neural_mab_algo.run_algorithm()
    print("Running Neural CoCoMaMa...")
    neural_cocoma_reward, neural_cocoma_regret, neural_cocoma_played_arms_arr, neural_cocoma_leaves_count_arr, neural_cocoma_metrics = neural_cocoma_algo.run_algorithm()
    print("Running CoCoMaMa...")
    omv_reward, omv_regret, omv_played_arms_arr, omv_leaves_count_arr, omv_metrics = omv_ucb_algo.run_algorithm()
    print("Running HD-ACC-UCB...")
    ucb_reward, ucb_regret, ucb_played_arms_arr, ucb_leaves_count_arr = cc_ucb_algo.run_algorithm()
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
        'cocomama_metrics': omv_metrics,
        'neural_cocoma_metrics': neural_cocoma_metrics
    }


def create_dataset_if_needed(config):
    """Create datasets if they don't exist and are needed."""
    from create_streaming_datasets import create_sprout_streaming_dataset, create_synthetic_from_streaming_sprout
    
    # Ensure datasets directory exists
    os.makedirs("datasets", exist_ok=True)
    
    if config["PROBLEM_MODEL_TYPE"] == "streaming_sprout":
        dataset_path = config.get("streaming_dataset_path", "datasets/sprout_streaming_300.arrow")
        if not os.path.exists(dataset_path):
            print(f"📦 Creating SPROUT streaming dataset: {dataset_path}")
            create_sprout_streaming_dataset(
                dataset_path,
                config["embedding_config"],
                config["num_rounds"],
                force_reload=False
            )
        else:
            print(f"✓ SPROUT streaming dataset already exists: {dataset_path}")
    
    elif config["PROBLEM_MODEL_TYPE"] == "streaming_synthetic":
        if config.get("create_dataset_if_missing", False):
            synthetic_path = config.get("synthetic_dataset_path", "datasets/synthetic_streaming_300.arrow")
            sprout_path = config.get("sprout_streaming_path", "datasets/sprout_streaming_300.arrow")
            
            # Create SPROUT dataset first if needed
            if not os.path.exists(sprout_path):
                print(f"📦 Creating SPROUT dataset: {sprout_path}")
                create_sprout_streaming_dataset(
                    sprout_path,
                    config["embedding_config"],
                    config["num_rounds"],
                    force_reload=False
                )
            else:
                print(f"✓ SPROUT dataset already exists: {sprout_path}")
            
            # Create synthetic dataset if needed
            if not os.path.exists(synthetic_path):
                print(f"📦 Creating synthetic dataset: {synthetic_path}")
                synthetic_config = config.get("synthetic_agents_config", {})
                create_synthetic_from_streaming_sprout(
                    sprout_path,
                    synthetic_path,
                    config["embedding_config"],
                    config["num_rounds"],
                    synthetic_config.get("specialized_agent_interval", 200),
                    synthetic_config.get("expert_start_time", 2000),
                    synthetic_config.get("expert_strength_breakpoint", 6000),
                    synthetic_config.get("num_expert_dimensions", 5),
                    synthetic_config.get("num_task_fit_dimensions", 1)
                )
            else:
                print(f"✓ Synthetic dataset already exists: {synthetic_path}")
            
            # Update config to use the synthetic dataset for streaming
            config["streaming_dataset_path"] = synthetic_path

def get_problem_model(config):
    """Create and return the appropriate problem model based on configuration."""
    # Create datasets if needed
    create_dataset_if_needed(config)
    
    if config["PROBLEM_MODEL_TYPE"] == "sprout":
        return SPROUTProblemModel(
            config["num_rounds"], 
            max(config["budgets"]), config["use_saved_data"], config["embedding_config"]
        )
    elif config["PROBLEM_MODEL_TYPE"] == "streaming_synthetic":
        # For synthetic, we now use the streaming model with the created dataset
        dataset_path = config.get("synthetic_dataset_path", "datasets/synthetic_streaming_300.arrow")
        return StreamingProblemModel(
            dataset_path, 
            config["embedding_config"]["dimensions"], 
            config["embedding_config"]["dimensions"], 
            config["num_rounds"], 
            max(config["budgets"])
        )
    elif config["PROBLEM_MODEL_TYPE"] == "streaming_sprout":
        # Load SPROUT streaming dataset
        dataset_path = config.get("streaming_dataset_path", "datasets/sprout_streaming_300.arrow")
        return StreamingProblemModel(
            dataset_path, config["embedding_config"]["dimensions"], config["embedding_config"]["dimensions"], config["num_rounds"], max(config["budgets"])
        )
    else:
        raise ValueError(f"Unknown problem model type: {config['PROBLEM_MODEL_TYPE']}")


def run_simulation(problem_model, budget, config, use_streaming=False):
    """Run the simulation for a given budget."""
    print(f"Doing budget {budget}...")
    print(f"Memory usage before parallel processing: {get_memory_usage():.2f} MB")
    
    if config["num_threads_to_use"] == -1:
        num_threads = int(multiprocessing.cpu_count()-1)
    else:
        num_threads = config["num_threads_to_use"]
    print(f"Running on {num_threads} threads")
    
    if use_streaming:
        # For streaming models, run in parallel with separate problem model instances
        print(f"Running streaming algorithms in parallel on {num_threads} threads")
        parallel_results = Parallel(n_jobs=num_threads)(
            delayed(run_one_try_streaming)(i, budget, config) 
            for i in tqdm(range(config["num_times_to_run"]))
        )
    else:
        parallel_results = Parallel(n_jobs=num_threads)(
            delayed(run_one_try_traditional)(problem_model, i, budget, config) 
            for i in tqdm(range(config["num_times_to_run"]))
        )
    
    print(f"Memory usage after parallel processing: {get_memory_usage():.2f} MB")
    
    # Save results
    results_file = f'{config["RESULTS_DIR"]}/parallel_results_budget_{budget}'
    if use_streaming:
        results_file += '_streaming'
    
    with open(results_file, 'wb') as output:
        pickle.dump(parallel_results, output, pickle.HIGHEST_PROTOCOL)
    
    return parallel_results


def generate_ablation_configs(ablation_config: dict) -> list[dict]:
    """
    Generate individual configs for each ablation parameter value.
    
    Args:
        ablation_config: Base configuration dictionary containing ablation parameters
    
    Returns:
        List of configuration dictionaries, one for each parameter value
    """
    configs = []
    base_config = ablation_config["base_config"].copy()
    
    # Merge base config with main config
    for key, value in base_config.items():
        if key not in ablation_config:
            ablation_config[key] = value
    
    for value in ablation_config["ablation_values"]:
        config = ablation_config.copy()
        config[ablation_config["ablation_parameter"]] = value
        config["ablation_variant_name"] = f"{ablation_config['ablation_algorithm']}_({ablation_config['ablation_parameter']}={value})"
        configs.append(config)
    
    return configs


def run_single_ablation_config(config: dict, variant_name: str) -> dict:
    """
    Run a single ablation configuration.
    
    Args:
        config: Configuration dictionary containing ablation parameters
        variant_name: Name of the variant being run (e.g., "streaming_neural_cocoma_(theta=2.0)")
    
    Returns:
        Dictionary containing results for all budgets
    """
    print(f"\n{'='*60}")
    print(f"🧪 Running ablation variant: {variant_name}")
    print(f"{'='*60}")
    
    # Extract parameter value from variant name for subfolder
    param_value = variant_name.split('=')[-1].rstrip(')')
    param_name = config["ablation_parameter"]
    variant_dir = os.path.join(config["RESULTS_DIR"], f"{param_name}_{param_value}")
    
    # Create variant-specific results directory
    os.makedirs(variant_dir, exist_ok=True)
    
    # Update config to use variant-specific directory
    variant_config = config.copy()
    variant_config["RESULTS_DIR"] = variant_dir
    
    # Create datasets if needed
    create_dataset_if_needed(variant_config)
    
    # Load problem model
    problem_model = get_problem_model(variant_config)
    
    # Run simulations for all budgets
    all_results = {}
    for budget in variant_config["budgets"]:
        print(f"\nRunning budget {budget} for {variant_name}...")
        budget_results = run_simulation(problem_model, budget, variant_config, use_streaming=True)
        all_results[budget] = budget_results
    
    return all_results


def run_ablation_study(ablation_config: dict) -> dict:
    """
    Run ablation study with multiple parameter values.
    
    This function runs the specified algorithm with different parameter values,
    saves results in separate subfolders, and generates comparison plots.
    
    Args:
        ablation_config: Configuration dictionary containing:
            - ablation_algorithm: Algorithm to test
            - ablation_parameter: Parameter to vary
            - ablation_values: List of parameter values to test
            - num_times_to_run: Number of runs per parameter value
            - RESULTS_DIR: Base results directory
            - Other experiment parameters
    
    Returns:
        Dictionary with aggregated results for plotting
    """
    print(f"\n🔬 Starting Ablation Study")
    print(f"Algorithm: {ablation_config['ablation_algorithm']}")
    print(f"Parameter: {ablation_config['ablation_parameter']}")
    print(f"Values: {ablation_config['ablation_values']}")
    print(f"Runs per value: {ablation_config['num_times_to_run']}")
    print(f"Results directory: {ablation_config['RESULTS_DIR']}")
    
    # Create main results directory
    os.makedirs(ablation_config["RESULTS_DIR"], exist_ok=True)
    
    # Check if we should only redo plots
    if ablation_config.get("only_redo_plots", False):
        print("📊 Only redoing plots with existing results...")
    else:
        # Generate configs for each parameter value
        configs = generate_ablation_configs(ablation_config)
        
        # Run each configuration
        for i, config in enumerate(configs):
            variant_name = config["ablation_variant_name"]
            print(f"\n📊 Processing variant {i+1}/{len(configs)}: {variant_name}")
            
            # Run the configuration (results saved to subfolder)
            run_single_ablation_config(config, variant_name)
    
    # Aggregate results for plotting
    print(f"\n📈 Aggregating results for plotting...")
    aggregated_results = aggregate_ablation_results(ablation_config)
    
    # Generate plots
    if ablation_config.get("plot", False):
        print(f"\n🎨 Generating ablation study plots...")
        generate_ablation_plots(aggregated_results, ablation_config)
    
    print(f"\n✅ Ablation study completed!")
    return aggregated_results


def aggregate_ablation_results(ablation_config: dict) -> dict:
    """
    Aggregate results from all ablation subfolders for plotting.
    
    This function reads the parallel_results_budget_X_streaming files from each
    parameter value subfolder and aggregates them across all runs.
    
    Args:
        ablation_config: Configuration dictionary containing ablation parameters
    
    Returns:
        Dictionary with aggregated results in format expected by plotting functions
    """
    import pickle
    import numpy as np
    
    results_dir = ablation_config["RESULTS_DIR"]
    ablation_algorithm = ablation_config["ablation_algorithm"]
    ablation_parameter = ablation_config["ablation_parameter"]
    ablation_values = ablation_config["ablation_values"]
    budgets = ablation_config["budgets"]
    
    print(f"📊 Aggregating results from subfolders...")
    
    # Initialize aggregated results structure
    aggregated = {}
    for budget in budgets:
        aggregated[budget] = {}
    
    # Collect data from all parameter value subfolders
    for param_value in ablation_values:
        variant_dir = os.path.join(results_dir, f"{ablation_parameter}_{param_value}")
        
        if not os.path.exists(variant_dir):
            print(f"⚠️  Warning: Subfolder {variant_dir} not found, skipping...")
            continue
            
        print(f"📁 Reading from {variant_dir}")
        
        # Read results for each budget
        for budget in budgets:
            results_file = os.path.join(variant_dir, f"parallel_results_budget_{budget}_streaming")
            
            if not os.path.exists(results_file):
                print(f"⚠️  Warning: Results file {results_file} not found, skipping...")
                continue
            
            # Load the results
            with open(results_file, 'rb') as f:
                budget_results = pickle.load(f)
            
            # Process each run in the budget results
            for run_data in budget_results:
                for key, value in run_data.items():
                    if 'reward' in key or 'regret' in key or 'leaves' in key:
                        # Create key with parameter value
                        if ablation_algorithm in key:
                            new_key = key.replace(ablation_algorithm, f'{ablation_algorithm}_{ablation_parameter}_{param_value}')
                        else:
                            new_key = key
                        
                        if new_key not in aggregated[budget]:
                            aggregated[budget][new_key] = []
                        aggregated[budget][new_key].append(value)
    
    # Process the aggregated results to create the expected format for plotting
    processed_aggregated = {}
    for budget in budgets:
        processed_aggregated[budget] = {}
        
        # Process each algorithm variant
        for key, values in aggregated[budget].items():
            if values:  # Only process if we have data
                # Calculate averages and standard deviations
                if 'reward' in key:
                    avg_key = key.replace('_reward', '_avg_reward')
                    std_key = key.replace('_reward', '_std_reward')
                elif 'regret' in key:
                    avg_key = key.replace('_regret', '_avg_regret')
                    std_key = key.replace('_regret', '_std_regret')
                elif 'leaves' in key:
                    avg_key = key.replace('_leaves_count_arr', '_avg_leaves')
                    std_key = key.replace('_leaves_count_arr', '_std_leaves')
                else:
                    continue
                
                # Calculate statistics
                if len(values) > 0:
                    # Convert to numpy arrays for easier calculation
                    values_array = np.array(values)
                    
                    if values_array.ndim == 2:  # Multiple runs
                        if 'reward' in key or 'regret' in key:
                            # For reward and regret, calculate cumulative averages first, then average across runs
                            cumulative_avgs = []
                            for run_idx in range(values_array.shape[0]):
                                run_data = values_array[run_idx]
                                if 'reward' in key:
                                    # For reward: cumulative sum / task number (average reward up to t)
                                    cumulative_avg = np.cumsum(run_data) / np.arange(1, len(run_data) + 1)
                                else:  # regret
                                    # For regret: cumulative sum (cumulative regret up to t)
                                    cumulative_avg = np.cumsum(run_data)
                                cumulative_avgs.append(cumulative_avg)
                            
                            # Average the cumulative values across runs
                            cumulative_avgs_array = np.array(cumulative_avgs)
                            processed_aggregated[budget][avg_key] = np.mean(cumulative_avgs_array, axis=0).tolist()
                            processed_aggregated[budget][std_key] = np.std(cumulative_avgs_array, axis=0).tolist()
                        else:
                            # For other metrics (like leaves), use simple averaging
                            processed_aggregated[budget][avg_key] = np.mean(values_array, axis=0).tolist()
                            processed_aggregated[budget][std_key] = np.std(values_array, axis=0).tolist()
                    else:  # Single run
                        if 'reward' in key or 'regret' in key:
                            # Calculate cumulative values for single run
                            if 'reward' in key:
                                cumulative_avg = np.cumsum(values_array) / np.arange(1, len(values_array) + 1)
                            else:  # regret
                                cumulative_avg = np.cumsum(values_array)
                            processed_aggregated[budget][avg_key] = cumulative_avg.tolist()
                            processed_aggregated[budget][std_key] = [0.0] * len(values_array)
                        else:
                            processed_aggregated[budget][avg_key] = values_array.tolist()
                            processed_aggregated[budget][std_key] = [0.0] * len(values_array)
    
    print(f"✅ Aggregated results from {len(ablation_values)} parameter values")
    return processed_aggregated


def generate_ablation_plots(aggregated_results: dict, config: dict) -> None:
    """
    Generate plots for ablation study results.
    
    Args:
        aggregated_results: Dictionary containing aggregated results for plotting
        config: Configuration dictionary containing plotting parameters
    """
    from plotting.ablation import create_ablation_plots
    
    results_dir = config["RESULTS_DIR"]
    num_rounds = config["num_rounds"]
    
    # Extract ablation study parameters
    algorithm_name = config["ablation_algorithm"]
    parameter_name = config["ablation_parameter"]
    parameter_values = config["ablation_values"]
    budgets = config["budgets"]
    
    # Generate all ablation plots using the improved plotting functions
    create_ablation_plots(
        aggregated_results=aggregated_results,
        budgets=budgets,
        num_rounds=num_rounds,
        results_dir=results_dir,
        algorithm_name=algorithm_name,
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        num_std_to_show=config.get("num_std_to_show", 5)
    )
    
    print(f"✅ Ablation plots saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run streaming contextual bandit experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (no arguments needed) - saves config automatically
  python main_streaming.py
  
  # Use config file (no interaction)
  python main_streaming.py --config_file config.yaml
  
  # Override config file with CLI and save result
  python main_streaming.py --config_file config.yaml --num_rounds 500 --save_config my_config.yaml
  
  # Run ablation study
  python main_streaming.py --ablation_config ablation_config.yaml
        """
    )
    
    # Config file option
    parser.add_argument("--config_file", type=str, default=None,
                       help="Path to YAML configuration file. If not provided, uses interactive mode.")
    
    # Ablation study option
    parser.add_argument("--ablation_config", type=str, default=None,
                       help="Path to ablation study configuration file.")
    
    # CLI override options (only used when config_file is provided)
    parser.add_argument("--problem_model_type", 
                       choices=["sprout", "streaming_sprout", "streaming_synthetic"], 
                       default=None,
                       help="Type of problem model to use (only when using --config_file)")
    
    parser.add_argument("--streaming_dataset_path", type=str, default=None,
                       help="Path to streaming Arrow dataset (only when using --config_file)")
    
    parser.add_argument("--sprout_streaming_path", type=str, default=None,
                       help="Path to SPROUT streaming dataset for synthetic generation (only when using --config_file)")
    
    parser.add_argument("--synthetic_dataset_path", type=str, default=None,
                       help="Path to synthetic dataset output (only when using --config_file)")
    
    parser.add_argument("--create_dataset_if_missing", default=None,action="store_true",
                       help="Create datasets if they don't exist (only when using --config_file)")
    
    parser.add_argument("--results_dir", type=str, default=None,
                       help="Results directory for output files (only when using --config_file)")
    
    parser.add_argument("--num_rounds", type=int, default=None,
                       help="Number of rounds to run (only when using --config_file)")
    
    parser.add_argument("--num_times_to_run", type=int, default=None,
                       help="Number of times to run each experiment (only when using --config_file)")
    
    parser.add_argument("--budgets", nargs='+', type=int, default=None,
                       help="List of budgets to test (only when using --config_file)")
    
    
    parser.add_argument("--v1", type=float, default=None,
                       help="V1 parameter for algorithms (only when using --config_file)")
    
    parser.add_argument("--v2", type=float, default=None,
                       help="V2 parameter for algorithms (only when using --config_file)")
    
    parser.add_argument("--rho", type=float, default=None,
                       help="Rho parameter for algorithms (only when using --config_file)")
    
    parser.add_argument("--N", type=int, default=None,
                       help="N parameter for algorithms (only when using --config_file)")
    
    # Synthetic dataset parameters
    parser.add_argument("--specialized_agent_interval", type=int, default=None,
                       help="Specialized agent interval (only when using --config_file)")
    
    parser.add_argument("--expert_start_time", type=int, default=None,
                       help="Expert start time (only when using --config_file)")
    
    parser.add_argument("--expert_strength_breakpoint", type=int, default=None,
                       help="Expert strength breakpoint (only when using --config_file)")
    
    parser.add_argument("--num_expert_dimensions", type=int, default=None,
                       help="Number of expert dimensions (only when using --config_file)")
    
    parser.add_argument("--num_task_fit_dimensions", type=int, default=None,
                       help="Number of task fit dimensions (only when using --config_file)")
    
    parser.add_argument("--embedding_model", type=str, default=None,
                       help="Embedding model name (only when using --config_file)")
    
    parser.add_argument("--embedding_dimensions", type=int, default=None,
                       help="Embedding dimensions (only when using --config_file)")
    
    parser.add_argument("--num_threads_to_use", type=int, default=None,
                       help="Number of threads to use (only when using --config_file)")
    
    parser.add_argument("--use_saved_data", action="store_true",
                       help="Use saved data instead of generating new (only when using --config_file)")
    
    parser.add_argument("--use_generated_workers_in_paper", action="store_true",
                       help="Use generated workers from paper (only when using --config_file)")
    
    parser.add_argument("--only_redo_plots", action="store_true",
                       help="Only redo plots using existing results (only when using --config_file)")
    
    parser.add_argument("--use_streaming_algorithms", action="store_true",
                       help="Use streaming algorithms instead of traditional ones (only when using --config_file)")
    
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots after running experiments (only when using --config_file)")
    
    parser.add_argument("--save_config", type=str, default=None,
                       help="Save the final configuration to a file (only when using --config_file)")
    
    args = parser.parse_args()
    
    # Handle ablation study
    if args.ablation_config:
        print(f"🔬 Loading ablation study configuration from: {args.ablation_config}")
        ablation_config = load_config_file(args.ablation_config)
        
        # Validate ablation config
        required_keys = ['ablation_algorithm', 'ablation_parameter', 'ablation_values', 'base_config']
        for key in required_keys:
            if key not in ablation_config:
                raise ValueError(f"Missing required key in ablation config: {key}")
        
        # Run ablation study
        run_ablation_study(ablation_config)
        return
    
    # Determine configuration mode
    if args.config_file:
        # Config file mode - no interaction
        print(f"📁 Loading configuration from: {args.config_file}")
        config_file = load_config_file(args.config_file)
        
        # Prepare CLI arguments for merging
        cli_args = {
            "PROBLEM_MODEL_TYPE": args.problem_model_type,
            "streaming_dataset_path": args.streaming_dataset_path,
            "sprout_streaming_path": args.sprout_streaming_path,
            "synthetic_dataset_path": args.synthetic_dataset_path,
            "create_dataset_if_missing": args.create_dataset_if_missing,
            "RESULTS_DIR": args.results_dir,
            "num_rounds": args.num_rounds,
            "num_times_to_run": args.num_times_to_run,
            "budgets": args.budgets,
            "v1": args.v1,
            "v2": args.v2,
            "rho": args.rho,
            "N": args.N,
            "num_threads_to_use": args.num_threads_to_use,
            "use_saved_data": args.use_saved_data,
            "use_generated_workers_in_paper": args.use_generated_workers_in_paper,
            "only_redo_plots": args.only_redo_plots
        }
        
        # Handle synthetic agents config separately
        synthetic_config = {}
        if args.specialized_agent_interval is not None:
            synthetic_config["specialized_agent_interval"] = args.specialized_agent_interval
        if args.expert_start_time is not None:
            synthetic_config["expert_start_time"] = args.expert_start_time
        if args.expert_strength_breakpoint is not None:
            synthetic_config["expert_strength_breakpoint"] = args.expert_strength_breakpoint
        if args.num_expert_dimensions is not None:
            synthetic_config["num_expert_dimensions"] = args.num_expert_dimensions
        if args.num_task_fit_dimensions is not None:
            synthetic_config["num_task_fit_dimensions"] = args.num_task_fit_dimensions
        
        if synthetic_config:
            cli_args["synthetic_agents_config"] = synthetic_config
        
        # Handle embedding config separately
        if args.embedding_model or args.embedding_dimensions:
            embedding_config = config_file.get("embedding_config", {}).copy()
            if args.embedding_model:
                embedding_config["model_name"] = args.embedding_model
            if args.embedding_dimensions:
                embedding_config["dimensions"] = args.embedding_dimensions
                # Update suffix to reflect new dimensions
                embedding_config["suffix"] = f"_{embedding_config['model_name']}_{args.embedding_dimensions}-dim"
            cli_args["embedding_config"] = embedding_config
        
        # Merge configurations
        config = merge_configs(config_file, cli_args)
        # Use streaming algorithms flag from config file, but allow CLI override
        use_streaming_algorithms = config.get("use_streaming_algorithms", args.use_streaming_algorithms)
        
        # Save configuration if requested
        if args.save_config:
            saved_file = save_config_to_file(config, use_streaming_algorithms, args.save_config)
            if saved_file:
                print(f"🔄 Configuration saved to: {saved_file}")
        
    else:
        # Interactive mode
        print("🎯 Starting interactive configuration mode...")
        print("   (Use --config_file to skip interactive mode)")
        config, use_streaming_algorithms = interactive_configuration()
    
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    # Create results directory
    os.makedirs(config["RESULTS_DIR"], exist_ok=True)
    
    # Print configuration summary
    print("\n" + "="*60)
    print("📋 CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Problem model type: {config['PROBLEM_MODEL_TYPE']}")
    print(f"Number of rounds: {config['num_rounds']}")
    print(f"Number of runs: {config['num_times_to_run']}")
    print(f"Budgets: {config['budgets']}")
    print(f"Results directory: {config['RESULTS_DIR']}")
    print(f"Embedding model: {config['embedding_config']['model_name']}")
    print(f"Embedding dimensions: {config['embedding_config']['dimensions']}")
    print(f"Use streaming algorithms: {use_streaming_algorithms}")
    print("="*60 + "\n")
    
    # Run experiment if not only redoing plots
    if not config["only_redo_plots"]:
        if not config["use_saved_data"] or config["PROBLEM_MODEL_TYPE"] == "streaming":
            # Create datasets once before all runs to ensure they exist
            print("Ensuring datasets are created...")
            create_dataset_if_needed(config)
            
            print("Loading problem model...")
            problem_model = get_problem_model(config)
            print(f"Memory usage after problem model initialization: {get_memory_usage():.2f} MB")
            
            # Check context space bounds
            print("\n=== Context Space Analysis ===")
            if config["PROBLEM_MODEL_TYPE"] == "streaming":
                # For streaming models, skip context analysis to avoid creating iterators
                # that would make the model non-pickleable
                print("Skipping context analysis for streaming models to maintain pickleability")
            else:
                sample_arms = problem_model.get_available_arms(1)
                if sample_arms:
                    sample_context = sample_arms[0].context
                    print(f"Context dimension: {len(sample_context)}")
                    print(f"Context min value: {np.min(sample_context):.6f}")
                    print(f"Context max value: {np.max(sample_context):.6f}")
                    print(f"Context mean value: {np.mean(sample_context):.6f}")
                    print(f"Context std value: {np.std(sample_context):.6f}")
                    print(f"Context L2 norm: {np.linalg.norm(sample_context):.6f}")
            print("=== End Context Space Analysis ===\n")
            
            # Run simulations
            for budget in config["budgets"]:
                run_simulation(problem_model, budget, config, use_streaming_algorithms)
        else:
            print("Skipping experiments (--only_redo_plots specified)")
    
    # Handle plotting
    if config.get("plot", False) or config.get("only_redo_plots", False):
        if use_streaming_algorithms:
            print("Loading and processing streaming results for plotting...")
            processed_data = load_and_process_streaming_results(config)
            
            # Define algorithm colors and styles to match the expected format
            algorithm_colors = {
                'HD-ACC-UCB': 'red',
                'CoCoMaMa': 'green', 
                'CC-MAB': 'blue',
                'Neural-CoCoMaMa (ours)': 'cyan',
                'Neural-MAB': 'orange',
                'Random': 'purple',
                'Oracle': 'black'
            }
            
            line_style_dict = {1: '--', 2: '-', 3: '-.', 4: ':'}
            num_std_to_show = config.get("num_std_to_show", 1)
            
            # Create plots for streaming algorithms using imported plotting functions
            print("Creating streaming algorithm plots...")
            num_rounds = len(processed_data[config["budgets"][0]]['ucb_avg_reward'])
            results_dir = config["RESULTS_DIR"]
            plot_all_cumulative_regret(processed_data, config["budgets"], num_rounds, num_std_to_show, algorithm_colors, results_dir)
            plot_all_average_reward(processed_data, config["budgets"], num_rounds, num_std_to_show, algorithm_colors, results_dir)
            plot_all_average_leaves(processed_data, config["budgets"], num_rounds, num_std_to_show, line_style_dict, algorithm_colors, results_dir)
            # Skip plot_additional_metrics for streaming as it expects different metrics structure
            print("Skipping additional metrics plot (not compatible with streaming algorithms)")
            
            # Plot selected agents if applicable
            if should_plot_selected_agents_streaming(config):
                plot_selected_agents(processed_data, config["num_rounds"], config["RESULTS_DIR"], get_agent_names_streaming)
            
            plt.show()
            print("Streaming plots created successfully!")
        else:
            print("Plotting is only supported for streaming algorithms. Enable streaming algorithms in your configuration.")
    
    print("Experiment completed!")


def load_and_process_streaming_results(config):
    """Load and process streaming results for all budgets, returning a dictionary of processed data."""
    all_processed_data = {}
    results_dir = config["RESULTS_DIR"]
    budgets = config["budgets"]
    
    # Determine number of rounds from the actual data
    with open(f'{results_dir}/parallel_results_budget_{budgets[0]}_streaming', 'rb') as input_file:
        sample_results = pickle.load(input_file)
    num_rounds = len(sample_results[0]['streaming_random_reward'])
    print(f"Detected {num_rounds} rounds in the data")
    
    for budget in budgets:
        # Load streaming results
        with open(f'{results_dir}/parallel_results_budget_{budget}_streaming', 'rb') as input_file:
            parallel_results = pickle.load(input_file)
        
        num_runs = len(parallel_results)
        
        # Initialize arrays for streaming algorithms
        streaming_random_reward_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_bench_reward_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_cocoma_reward_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_neural_cocoma_reward_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_hd_acc_ucb_reward_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_cc_mab_reward_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_neural_mab_reward_runs_arr = np.zeros((num_runs, num_rounds))
        
        streaming_random_regret_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_bench_regret_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_cocoma_regret_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_neural_cocoma_regret_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_hd_acc_ucb_regret_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_cc_mab_regret_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_neural_mab_regret_runs_arr = np.zeros((num_runs, num_rounds))
        
        streaming_cocoma_leaves_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_neural_cocoma_leaves_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_hd_acc_ucb_leaves_runs_arr = np.zeros((num_runs, num_rounds))
        
        for i, entry in enumerate(parallel_results):
            # Process rewards
            streaming_random_reward_runs_arr[i] = pd.Series(entry['streaming_random_reward']).expanding().mean().values
            streaming_bench_reward_runs_arr[i] = pd.Series(entry['streaming_bench_reward']).expanding().mean().values
            streaming_cocoma_reward_runs_arr[i] = pd.Series(entry['streaming_cocoma_reward']).expanding().mean().values
            streaming_neural_cocoma_reward_runs_arr[i] = pd.Series(entry['streaming_neural_cocoma_reward']).expanding().mean().values
            streaming_hd_acc_ucb_reward_runs_arr[i] = pd.Series(entry['streaming_hd_acc_ucb_reward']).expanding().mean().values
            streaming_cc_mab_reward_runs_arr[i] = pd.Series(entry['streaming_cc_mab_reward']).expanding().mean().values
            streaming_neural_mab_reward_runs_arr[i] = pd.Series(entry['streaming_neural_mab_reward']).expanding().mean().values
            
            # Process regrets (benchmark doesn't have regret as it's the oracle)
            streaming_random_regret_runs_arr[i] = np.cumsum(entry['streaming_random_regret'])
            streaming_bench_regret_runs_arr[i] = np.zeros(num_rounds)  # Benchmark has no regret
            streaming_cocoma_regret_runs_arr[i] = np.cumsum(entry['streaming_cocoma_regret'])
            streaming_neural_cocoma_regret_runs_arr[i] = np.cumsum(entry['streaming_neural_cocoma_regret'])
            streaming_hd_acc_ucb_regret_runs_arr[i] = np.cumsum(entry['streaming_hd_acc_ucb_regret'])
            streaming_cc_mab_regret_runs_arr[i] = np.cumsum(entry['streaming_cc_mab_regret'])
            streaming_neural_mab_regret_runs_arr[i] = np.cumsum(entry['streaming_neural_mab_regret'])
            
            # Process leaves (these are already the count of leaves at each round, no need for expanding mean)
            streaming_cocoma_leaves_runs_arr[i] = np.array(entry['streaming_cocoma_leaves_count_arr'])
            streaming_neural_cocoma_leaves_runs_arr[i] = np.array(entry['streaming_neural_cocoma_leaves_count_arr'])
            streaming_hd_acc_ucb_leaves_runs_arr[i] = np.array(entry['streaming_hd_acc_ucb_leaves_count_arr'])
            
            # Add neural_cocoma_metrics to the entry for compatibility with plotting functions
            if 'streaming_neural_cocoma_metrics' in entry:
                entry['neural_cocoma_metrics'] = entry['streaming_neural_cocoma_metrics']
        
        all_processed_data[budget] = {
            # Map streaming algorithms to expected key names for plotting functions
            'ucb_avg_reward': np.mean(streaming_hd_acc_ucb_reward_runs_arr, axis=0),
            'cocomama_avg_reward': np.mean(streaming_cocoma_reward_runs_arr, axis=0),
            'cc_mab_avg_reward': np.mean(streaming_cc_mab_reward_runs_arr, axis=0),
            'neural_cocoma_avg_reward': np.mean(streaming_neural_cocoma_reward_runs_arr, axis=0),
            'neural_mab_avg_reward': np.mean(streaming_neural_mab_reward_runs_arr, axis=0),
            'random_avg_reward': np.mean(streaming_random_reward_runs_arr, axis=0),
            'bench_avg_reward': np.mean(streaming_bench_reward_runs_arr, axis=0),
            
            'ucb_std_reward': np.std(streaming_hd_acc_ucb_reward_runs_arr, axis=0),
            'cocomama_std_reward': np.std(streaming_cocoma_reward_runs_arr, axis=0),
            'cc_mab_std_reward': np.std(streaming_cc_mab_reward_runs_arr, axis=0),
            'neural_cocoma_std_reward': np.std(streaming_neural_cocoma_reward_runs_arr, axis=0),
            'neural_mab_std_reward': np.std(streaming_neural_mab_reward_runs_arr, axis=0),
            'random_std_reward': np.std(streaming_random_reward_runs_arr, axis=0),
            'bench_std_reward': np.std(streaming_bench_reward_runs_arr, axis=0),
            
            'ucb_avg_regret': np.mean(streaming_hd_acc_ucb_regret_runs_arr, axis=0),
            'cocomama_avg_regret': np.mean(streaming_cocoma_regret_runs_arr, axis=0),
            'cc_mab_avg_regret': np.mean(streaming_cc_mab_regret_runs_arr, axis=0),
            'neural_cocoma_avg_regret': np.mean(streaming_neural_cocoma_regret_runs_arr, axis=0),
            'neural_mab_avg_regret': np.mean(streaming_neural_mab_regret_runs_arr, axis=0),
            'random_avg_regret': np.mean(streaming_random_regret_runs_arr, axis=0),
            
            'ucb_std_regret': np.std(streaming_hd_acc_ucb_regret_runs_arr, axis=0),
            'cocomama_std_regret': np.std(streaming_cocoma_regret_runs_arr, axis=0),
            'cc_mab_std_regret': np.std(streaming_cc_mab_regret_runs_arr, axis=0),
            'neural_cocoma_std_regret': np.std(streaming_neural_cocoma_regret_runs_arr, axis=0),
            'neural_mab_std_regret': np.std(streaming_neural_mab_regret_runs_arr, axis=0),
            'random_std_regret': np.std(streaming_random_regret_runs_arr, axis=0),
            
            'ucb_avg_leaves': np.mean(streaming_hd_acc_ucb_leaves_runs_arr, axis=0),
            'cocomama_avg_leaves': np.mean(streaming_cocoma_leaves_runs_arr, axis=0),
            'neural_cocoma_avg_leaves': np.mean(streaming_neural_cocoma_leaves_runs_arr, axis=0),
            'ucb_std_leaves': np.std(streaming_hd_acc_ucb_leaves_runs_arr, axis=0),
            'cocomama_std_leaves': np.std(streaming_cocoma_leaves_runs_arr, axis=0),
            'neural_cocoma_std_leaves': np.std(streaming_neural_cocoma_leaves_runs_arr, axis=0),
            'parallel_results': parallel_results  # Keep raw results for other plots
        }
    return all_processed_data


def should_plot_selected_agents_streaming(config):
    """Determine if we should plot selected agents for streaming models."""
    return config["PROBLEM_MODEL_TYPE"] == "streaming"  # Plot for streaming models


def get_agent_names_streaming():
    """Get the list of agent names for streaming models."""
    # For streaming models, we don't have fixed agent names
    return None





if __name__ == '__main__':
    main()
