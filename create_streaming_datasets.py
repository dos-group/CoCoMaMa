"""
Create streaming datasets directly from HuggingFace and generate synthetic datasets.
This module provides functionality to:
1. Create SPROUT streaming dataset directly from HuggingFace
2. Create synthetic agents dataset based on streaming SPROUT dataset
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import argparse

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from streaming_dataset import StreamingDataset


def load_sprout_from_huggingface(force_reload: bool = False) -> pd.DataFrame:
    """
    Load SPROUT dataset directly from HuggingFace.
    
    Args:
        force_reload: If True, reload from HuggingFace even if local file exists
        
    Returns:
        Pandas DataFrame with SPROUT data
    """
    saved_df_filename = 'sprout_df.pkl'
    
    if not force_reload and os.path.exists(saved_df_filename):
        print("Loading SPROUT dataset from local cache...")
        with open(saved_df_filename, 'rb') as f:
            return pd.read_pickle(f)
    
    print("Loading SPROUT dataset from HuggingFace...")
    ds = load_dataset("CARROT-LLM-Routing/SPROUT")
    
    # Convert to pandas DataFrame
    df = ds['train'].to_pandas()
    
    # Save for future use
    print(f"Saving SPROUT dataset to {saved_df_filename}...")
    with open(saved_df_filename, 'wb') as f:
        df.to_pickle(f)
    
    return df


def create_sprout_streaming_dataset(
    output_path: str,
    embedding_config: Dict[str, Any],
    num_rounds: Optional[int] = None,
    force_reload: bool = False
) -> str:
    """
    Create SPROUT streaming dataset directly from HuggingFace.
    
    Args:
        output_path: Path to save the Arrow streaming dataset
        embedding_config: Configuration for embeddings
        num_rounds: Number of rounds to include (None for all)
        force_reload: If True, reload from HuggingFace even if local file exists
        
    Returns:
        Path to the created streaming dataset
    """
    print("Creating SPROUT streaming dataset from HuggingFace...")
    
    # Load SPROUT data
    df = load_sprout_from_huggingface(force_reload)
    
    # Limit number of rounds if specified
    if num_rounds is not None:
        df = df.head(num_rounds)
        print(f"Limited to {num_rounds} rounds")
    
    print(f"Processing {len(df)} tasks...")
    
    # Initialize embedding model
    print(f"Loading embedding model: {embedding_config['model_name']}...")
    model = SentenceTransformer(
        embedding_config['model_name'],
        device='cpu',
        trust_remote_code=True,
        truncate_dim=embedding_config['dimensions'],
        cache_folder='./cache'
    )
    
    # Agent keys
    agent_keys = [
        "aws-claude-3-5-sonnet-v1",
        "aws-titan-text-premier-v1", 
        "openai-gpt-4o",
        "openai-gpt-4o-mini",
        "wxai-granite-3-2b-instruct-8k-max-tokens",
        "wxai-granite-3-8b-instruct-8k-max-tokens",
        "wxai-llama-3-1-70b-instruct",
        "wxai-llama-3-1-8b-instruct",
        "wxai-llama-3-2-1b-instruct",
        "wxai-llama-3-2-3b-instruct",
        "wxai-llama-3-3-70b-instruct",
        "wxai-llama-3-405b-instruct",
        "wxai-mixtral-8x7b-instruct-v01"
    ]
    
    # Create task embeddings
    print("Creating task embeddings...")
    task_embeddings = []
    for idx, prompt in enumerate(tqdm(df['prompt'], desc="Task embeddings")):
        embedding = model.encode(
            prompt,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        task_embeddings.append(embedding)
    
    # Create agent embeddings
    print("Creating agent embeddings...")
    agent_embeddings = {}
    for key in tqdm(agent_keys, desc="Agent embeddings"):
        # Load model card
        model_card_path = f'model_cards/{key}.json'
        if not os.path.exists(model_card_path):
            print(f"Warning: Model card not found at {model_card_path}")
            continue
            
        with open(model_card_path, 'r') as f:
            agent_card = json.load(f)
            agent_card_str = json.dumps(agent_card)
            agent_embedding = model.encode(
                agent_card_str,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            agent_embeddings[key] = agent_embedding
    
    # Create streaming tasks
    print("Creating streaming tasks...")
    tasks = []
    
    for idx, row in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing tasks")):
        _, data = row
        
        # Get task embedding
        task_embedding = task_embeddings[idx]
        
        # Create agents for this task
        agents = []
        for agent_idx, key in enumerate(agent_keys):
            if key not in agent_embeddings:
                continue
                
            # Get agent response data
            agent_data = data.get(key, {})
            if isinstance(agent_data, dict):
                rating = float(agent_data.get('score', 0.0))
            else:
                rating = 0.0
            
            agents.append({
                'agent_id': agent_idx,
                'agent_embedding': agent_embeddings[key],
                'rating': rating
            })
        
        tasks.append({
            'task_id': idx,
            'seq': idx,
            'task_embedding': task_embedding,
            'agents': agents
        })
    
    # Create streaming dataset
    print(f"Writing streaming dataset to {output_path}...")
    streaming_dataset = StreamingDataset(
        task_embedding_dim=embedding_config['dimensions'],
        agent_embedding_dim=embedding_config['dimensions']
    )
    streaming_dataset.write_dataset(tasks, output_path)
    
    print(f"✓ SPROUT streaming dataset created: {output_path}")
    print(f"  - {len(tasks)} tasks")
    print(f"  - {len(agent_keys)} agents per task")
    print(f"  - {embedding_config['dimensions']}D embeddings")
    
    return output_path


def create_synthetic_from_streaming_sprout(
    sprout_streaming_path: str,
    output_path: str,
    embedding_config: Dict[str, Any],
    num_rounds: int,
    specialized_agent_interval: int = 200,
    expert_start_time: int = 2000,
    expert_strength_breakpoint: int = 6000,
    num_expert_dimensions: int = 5,
    num_task_fit_dimensions: int = 1
) -> str:
    """
    Create synthetic agents dataset based on streaming SPROUT dataset.
    This follows the exact logic from SPROUTSyntheticAgentsProblemModel.
    
    Args:
        sprout_streaming_path: Path to the SPROUT streaming dataset
        output_path: Path to save the synthetic streaming dataset
        embedding_config: Configuration for embeddings
        num_rounds: Number of rounds to generate
        specialized_agent_interval: Interval for specialized agents
        expert_start_time: Start time for experts
        expert_strength_breakpoint: Breakpoint for expert strength
        num_expert_dimensions: Number of expert dimensions
        num_task_fit_dimensions: Number of task fit dimensions
        
    Returns:
        Path to the created synthetic streaming dataset
    """
    print("Creating synthetic agents dataset from streaming SPROUT...")
    
    # Load SPROUT streaming dataset
    sprout_dataset = StreamingDataset(
        embedding_config['dimensions'], 
        embedding_config['dimensions']
    )
    
    # Get SPROUT tasks for reference
    sprout_tasks = list(sprout_dataset.stream_tasks(sprout_streaming_path))
    print(f"Loaded {len(sprout_tasks)} SPROUT tasks for reference")
    
    # Initialize embedding model for synthetic data
    print(f"Loading embedding model: {embedding_config['model_name']}...")
    model = SentenceTransformer(
        embedding_config['model_name'],
        device='cpu',
        trust_remote_code=True,
        truncate_dim=embedding_config['dimensions'],
        cache_folder='./cache'
    )
    
    # Agent keys (same as original)
    agent_keys = [
        "aws-claude-3-5-sonnet-v1",
        "aws-titan-text-premier-v1",
        "openai-gpt-4o",
        "openai-gpt-4o-mini",
        "wxai-granite-3-2b-instruct-8k-max-tokens",
        "wxai-granite-3-8b-instruct-8k-max-tokens",
        "wxai-llama-3-1-70b-instruct",
        "wxai-llama-3-1-8b-instruct",
        "wxai-llama-3-2-1b-instruct",
        "wxai-llama-3-2-3b-instruct",
        "wxai-llama-3-3-70b-instruct",
        "wxai-llama-3-405b-instruct",
        "wxai-mixtral-8x7b-instruct-v01"
    ]
    
    # Specialized agents base candidates (same as original)
    specialized_agents_base_candidates = agent_keys.copy()
    
    # Create agent embeddings (same as original)
    print("Creating agent embeddings...")
    agent_embeddings_map = {}
    for key in tqdm(agent_keys, desc="Agent embeddings"):
        model_card_path = f'model_cards/{key}.json'
        if not os.path.exists(model_card_path):
            print(f"Warning: Model card not found at {model_card_path}")
            # Create a random embedding as fallback
            agent_embeddings_map[key] = np.random.randn(embedding_config['dimensions']).astype(np.float32)
            agent_embeddings_map[key] = agent_embeddings_map[key] / np.linalg.norm(agent_embeddings_map[key])
            continue
            
        with open(model_card_path, 'r') as f:
            agent_card = json.load(f)
            agent_card_str = json.dumps(agent_card)
            agent_embedding = model.encode(
                agent_card_str,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            agent_embeddings_map[key] = agent_embedding
    
    # Track specialized agents and used specializations (same as original)
    specialized_agents = []
    used_weak_specializations = set()
    used_strong_specializations = set()
    specializable_dims = set(range((embedding_config['dimensions'] - 1) // 2))
    
    # Create synthetic tasks
    print(f"Creating {num_rounds} synthetic tasks...")
    synthetic_tasks = []
    
    for t in tqdm(range(1, num_rounds + 1), desc="Creating synthetic tasks"):
        # Create synthetic task embedding (based on SPROUT patterns)
        if t - 1 < len(sprout_tasks):
            # Use real SPROUT task embedding as base
            task_embedding = sprout_tasks[t - 1]['task_embedding']
        else:
            # Generate random task embedding
            task_embedding = np.random.randn(embedding_config['dimensions']).astype(np.float32)
        
        # Normalize task embedding
        task_embedding = task_embedding / np.linalg.norm(task_embedding)
        
        # Create agents for this task
        agents = []
        agent_id = 0
        
        # Add general purpose agents (same as original)
        for available_agent in agent_keys:
            # Create context by concatenating task and agent embeddings
            context = np.concatenate([task_embedding, agent_embeddings_map[available_agent]])
            
            # Get base score from SPROUT dataset (use corresponding task if available)
            if t - 1 < len(sprout_tasks):
                # Find the corresponding agent in the SPROUT task
                sprout_task = sprout_tasks[t - 1]
                sprout_agent = None
                for sprout_agent_data in sprout_task['agents']:
                    if sprout_agent_data['agent_id'] == agent_keys.index(available_agent):
                        sprout_agent = sprout_agent_data
                        break
                
                if sprout_agent is not None:
                    base_score = sprout_agent['rating']
                else:
                    # Fallback to random if agent not found
                    base_score = np.random.uniform(0.0, 0.0)
            else:
                # Use random score for tasks beyond SPROUT dataset
                base_score = np.random.uniform(0.0, 0.0)
            
            # Calculate true mean using the original logic
            true_mean = calculate_specialized_task_true_mean(
                base_score,
                context,
                agent_embeddings_map[available_agent],
                False,  # Not a specialized agent
                num_task_fit_dimensions
            )
            
            agents.append({
                'agent_id': agent_id,
                'agent_embedding': agent_embeddings_map[available_agent],
                'rating': true_mean
            })
            agent_id += 1
        
        # Create new specialized agent (same as original logic)
        if t % specialized_agent_interval == 0 and t > expert_start_time:
            max_attempts = 1
            specialized_agent = None
            
            for _ in range(max_attempts):
                base_agent = np.random.choice(specialized_agents_base_candidates)
                available_dims = specializable_dims.copy()
                
                if len(available_dims) >= num_expert_dimensions:
                    expert_dims = sorted(np.random.choice(list(available_dims), num_expert_dimensions, replace=False))
                    expert_dims_tuple = tuple(expert_dims)
                    
                    # Determine if this will be a strong or weak expert based on time
                    is_strong_expert = t > expert_strength_breakpoint
                    used_specializations = used_strong_specializations if is_strong_expert else used_weak_specializations
                    
                    # Check if this exact combination has been used with this base agent
                    if (base_agent, expert_dims_tuple) not in used_specializations:
                        specialized_agent = {}
                        specialized_agent["base_agent"] = base_agent
                        specialized_agent["expert_dims"] = expert_dims
                        base_embedding = agent_embeddings_map[base_agent].copy()
                        
                        # Set all expert dimensions based on whether it's a strong or weak expert
                        for dim in expert_dims:
                            if is_strong_expert:
                                base_embedding[dim] = 1.0
                            else:
                                base_embedding[dim] = 0.9
                        
                        specialized_agent["agent_context"] = base_embedding
                        specialized_agents.append(specialized_agent)
                        used_specializations.add((base_agent, expert_dims_tuple))
                        break
            
            if specialized_agent is None:
                print(f"Warning: Could not create new specialized agent at time {t} after {max_attempts} attempts")
        
        # Add arms for specialized agents (same as original logic)
        for agent in specialized_agents:
            # Get the task context from the base agent's context
            task_context = task_embedding
            context = np.concatenate([task_embedding, agent["agent_context"]])
            
            # Get base score from SPROUT dataset for the base agent
            if t - 1 < len(sprout_tasks):
                # Find the corresponding base agent in the SPROUT task
                sprout_task = sprout_tasks[t - 1]
                sprout_agent = None
                base_agent_index = agent_keys.index(agent["base_agent"])
                for sprout_agent_data in sprout_task['agents']:
                    if sprout_agent_data['agent_id'] == base_agent_index:
                        sprout_agent = sprout_agent_data
                        break
                
                if sprout_agent is not None:
                    base_agent_score = sprout_agent['rating']
                else:
                    # Fallback to random if agent not found
                    base_agent_score = np.random.uniform(0.0, 0.0)
            else:
                # Use random score for tasks beyond SPROUT dataset
                base_agent_score = np.random.uniform(0.0, 0.0)
            
            # Calculate true mean using the helper function
            true_mean = calculate_specialized_task_true_mean(
                base_agent_score,
                context,
                agent["agent_context"],
                True,  # Is a specialized agent
                num_task_fit_dimensions
            )
            
            agents.append({
                'agent_id': agent_id,
                'agent_embedding': agent["agent_context"],
                'rating': true_mean
            })
            agent_id += 1
        
        synthetic_tasks.append({
            'task_id': t - 1,  # 0-indexed for consistency
            'seq': t - 1,
            'task_embedding': task_embedding,
            'agents': agents
        })
    
    # Create streaming dataset
    print(f"Writing synthetic streaming dataset to {output_path}...")
    synthetic_dataset = StreamingDataset(
        task_embedding_dim=embedding_config['dimensions'],
        agent_embedding_dim=embedding_config['dimensions']
    )
    synthetic_dataset.write_dataset(synthetic_tasks, output_path)
    
    print(f"✓ Synthetic streaming dataset created: {output_path}")
    print(f"  - {len(synthetic_tasks)} tasks")
    print(f"  - {len(agent_keys)} base agents per task")
    print(f"  - {len(specialized_agents)} specialized agents created")
    print(f"  - {embedding_config['dimensions']}D embeddings")
    
    return output_path


def calculate_specialized_task_true_mean(
    base_agent_score: float, 
    task_context: np.ndarray, 
    agent_context: np.ndarray, 
    is_specialized_agent: bool = True,
    num_task_fit_dimensions: int = 1
) -> float:
    """
    Calculate the true mean for an agent based on the original logic from SPROUTSyntheticAgentsProblemModel.
    This is an exact copy of the original calculate_specialized_task_true_mean method.
    
    Args:
        base_agent_score: float, the score of the base agent
        task_context: numpy array, the task embedding (first half of concatenated context)
        agent_context: numpy array, the agent embedding
        is_specialized_agent: bool, whether this is a specialized agent
        num_task_fit_dimensions: int, number of task fit dimensions
        
    Returns:
        float: The calculated true mean between 0 and 1
    """
    # Get the task embedding (first half of the concatenated context)
    task_embedding = task_context[:len(task_context)//2]
    
    # Find the dimensions with highest scalar values in task embedding
    top_dims = np.argsort(task_embedding)[-num_task_fit_dimensions:]
    
    # Calculate specialized component based on the top dimensions
    specialized_component = sum(task_embedding[dim] * (agent_context[dim]**100) for dim in top_dims) / num_task_fit_dimensions
    
    # Scale the specialized component to make sigmoid more meaningful
    scaled_component = 5.0 * specialized_component
    normalized_specialized = 1 / (1 + np.exp(-scaled_component))

    base_contribution = base_agent_score

    if (normalized_specialized < 0.6 or not is_specialized_agent):
        normalized_specialized = 0
        if(is_specialized_agent):
            base_contribution = base_contribution * 0.1
    # Combine base agent score (20%) with specialized component (80%)
    true_mean = 0.2 * base_contribution + 0.8 * normalized_specialized
    
    return true_mean


def main():
    parser = argparse.ArgumentParser(description="Create streaming datasets from HuggingFace")
    parser.add_argument("--dataset_type", choices=["sprout", "synthetic"], required=True,
                       help="Type of dataset to create")
    parser.add_argument("--output_path", required=True,
                       help="Output path for streaming dataset")
    parser.add_argument("--num_rounds", type=int, default=300,
                       help="Number of rounds")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--embedding_dimensions", type=int, default=384,
                       help="Embedding dimensions")
    parser.add_argument("--force_reload", action="store_true",
                       help="Force reload from HuggingFace")
    parser.add_argument("--sprout_streaming_path", 
                       help="Path to SPROUT streaming dataset (required for synthetic)")
    
    # Synthetic dataset parameters
    parser.add_argument("--specialized_agent_interval", type=int, default=200,
                       help="Interval for specialized agents")
    parser.add_argument("--expert_start_time", type=int, default=2000,
                       help="Start time for experts")
    parser.add_argument("--expert_strength_breakpoint", type=int, default=6000,
                       help="Breakpoint for expert strength")
    parser.add_argument("--num_expert_dimensions", type=int, default=5,
                       help="Number of expert dimensions")
    parser.add_argument("--num_task_fit_dimensions", type=int, default=1,
                       help="Number of task fit dimensions")
    
    args = parser.parse_args()
    
    embedding_config = {
        'model_name': args.embedding_model,
        'dimensions': args.embedding_dimensions,
        'suffix': f'_{args.embedding_model}_{args.embedding_dimensions}-dim'
    }
    
    if args.dataset_type == "sprout":
        create_sprout_streaming_dataset(
            args.output_path,
            embedding_config,
            args.num_rounds,
            args.force_reload
        )
    elif args.dataset_type == "synthetic":
        if not args.sprout_streaming_path:
            raise ValueError("--sprout_streaming_path is required for synthetic dataset creation")
        
        create_synthetic_from_streaming_sprout(
            args.sprout_streaming_path,
            args.output_path,
            embedding_config,
            args.num_rounds,
            args.specialized_agent_interval,
            args.expert_start_time,
            args.expert_strength_breakpoint,
            args.num_expert_dimensions,
            args.num_task_fit_dimensions
        )


if __name__ == "__main__":
    main()
