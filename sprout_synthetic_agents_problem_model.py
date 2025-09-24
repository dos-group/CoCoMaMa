import pickle
import random
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
import os

import sprout_loader
from Arm import Arm
from ProblemModel import ProblemModel
from Reward import Reward

"""
This file contains code for the SPROUT problem model. The SPROUTProblemModel class has functions to 
provide available arms, play the arms, calculate regret, and etc.
"""

saved_df_name = 'sprout_synthetic_agents_simulation_df.pkl'  # file where the saved simulation-ready dataframe will be saved
embeddings_df_name = 'sprout_synthetic_agents_embeddings_df.pkl'  # file where the embeddings will be saved

# Set up cache directory in the virtual environment
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cache')
os.makedirs(cache_dir, exist_ok=True)

def log_progress(current, total, prefix=""):
    """Log progress for every 1% completed"""
    percentage = (current / total) * 100
    if int(percentage) % 1 == 0 and percentage > 0:
        print(f"{prefix}Progress: {percentage:.1f}% ({current}/{total})")

class SPROUTSyntheticAgentsProblemModel(ProblemModel):
    def __init__(self, num_rounds, budget, use_saved, embedding_config=None, specialized_agent_interval=100, num_expert_dimensions=2, num_task_fit_dimensions=1, expert_start_time=1000, expert_strength_breakpoint=2000):
        self.num_rounds = num_rounds
        self.specialized_agent_interval = specialized_agent_interval
        self.num_expert_dimensions = num_expert_dimensions
        self.num_task_fit_dimensions = num_task_fit_dimensions
        self.expert_start_time = expert_start_time
        self.expert_strength_breakpoint = expert_strength_breakpoint
        self.embedding_config = embedding_config or {
            'model_name': 'all-MiniLM-L6-v2',
            'dimensions': 384,
            'suffix': ''
        }
        
        # Initialize the sentence transformer model
        print(f"Loading embedding model: {self.embedding_config['model_name']}...")
        self.model = SentenceTransformer(
            self.embedding_config['model_name'],
            device='cpu',
            trust_remote_code=True,
            truncate_dim=self.embedding_config['dimensions'],
            cache_folder=cache_dir
        )
        
        if not use_saved:
            self.df = self.initialize_df()
            self.df.set_index('time', inplace=True)
            with open(saved_df_name, 'wb') as output:
                pickle.dump(self.df, output, pickle.HIGHEST_PROTOCOL)
        else:
            with open(saved_df_name, 'rb') as input_file:
                self.df = pickle.load(input_file)

    def get_size(self):
        """
        Calculate the total number of arms across all time steps.
        Returns:
            int: Total number of arms in the problem model
        """
        return len(self.df)

    def get_available_arms(self, t):
        # Construct a list of Arm objects
        arm_list = []
        for _, row in self.df.loc[t].iterrows():
            arm_list.append(Arm(len(arm_list), row['context'], row['true_mean']))
        return arm_list

    def get_regret(self, t, budget, slate):
        df = self.df.loc[t]
        # TODO incorporate budget
        # Get all possible rewards for this timestep
        true_all_rewards = [Reward(Arm(i, row['context'], row['true_mean']), row['true_mean']) 
                      for i, (_, row) in enumerate(df.iterrows())]
        true_highest_possible_reward = self.get_total_reward(true_all_rewards)
        
        # Get rewards for selected slate
        true_slate_rewards = [Reward(arm, df.iloc[arm.unique_id]['true_mean']) for arm in slate]
        true_highest_slate_reward = self.get_total_reward(true_slate_rewards)

        return true_highest_possible_reward - true_highest_slate_reward

    def get_total_reward(self, rewards):
        # TODO: try other reward models
        max_reward = max(reward.quality for reward in rewards)
        return max_reward

    def play_arms(self, t, slate):
        reward_list = []
        df = self.df.loc[t]
        for arm in slate:
            # TODO: use LLM as judge to observe quality
            quality = df.iloc[arm.unique_id]['true_mean']
            reward_list.append(Reward(arm, quality))
        return reward_list

    def oracle(self, K, g_list):
        g_list = np.array(g_list)
        # Add a small random tie-breaker
        noise = np.random.rand(len(g_list)) * 1e-8
        randomized_g_list = g_list + noise
        return np.argsort(randomized_g_list)[-K:]

    def normalize_embedding(self, embedding):
        """
        Normalize an embedding vector to unit length.
        
        Args:
            embedding: numpy array of embedding vector
            
        Returns:
            Normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def combine_embeddings(self, task_embedding, agent_embedding, operation='cosine'):
        """
        Combine task and agent embeddings using various meaningful operations.
        
        Args:
            task_embedding: numpy array of task embedding
            agent_embedding: numpy array of agent card embedding
            operation: One of ['cosine', 'concatenate', 'hadamard', 'sum', 'subtract']
            
        Returns:
            Combined embedding vector
        """
        # Ensure embeddings are normalized
        task_embedding = self.normalize_embedding(task_embedding)
        agent_embedding = self.normalize_embedding(agent_embedding)
        
        if operation == 'cosine':
            # Cosine similarity between task and agent
            return np.dot(task_embedding, agent_embedding)
        elif operation == 'concatenate':
            # Concatenate the embeddings to preserve all information
            return np.concatenate([task_embedding, agent_embedding])
        elif operation == 'hadamard':
            # Element-wise multiplication (Hadamard product)
            return task_embedding * agent_embedding
        elif operation == 'sum':
            # Sum of embeddings
            return task_embedding + agent_embedding
        elif operation == 'subtract':
            # Difference between task and agent embeddings
            return task_embedding - agent_embedding
        else:
            raise ValueError("Operation must be one of ['cosine', 'concatenate', 'hadamard', 'sum', 'subtract']")

    def create_agent_embeddings(self, agent_keys):
        """
        Create embeddings for all agents using their model cards.
        
        Args:
            agent_keys: List of agent keys to create embeddings for
            
        Returns:
            dict: Mapping of agent keys to their embeddings
        """
        agent_embeddings_map = {}
        
        for key in agent_keys:
            print(f"Creating embedding for {key}...")
            try:
                # Load the model card JSON file for this agent
                with open(f'model_cards/{key}.json', 'r') as f:
                    agent_card = json.load(f)
                    # Convert JSON to string before encoding
                    agent_card_str = json.dumps(agent_card)
                    agent_embedding = self.model.encode(
                        agent_card_str,
                        convert_to_tensor=False,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    agent_embeddings_map[key] = agent_embedding
            except FileNotFoundError:
                print(f"Warning: Model card for {key} not found. Using random embedding.")
                # Create a random embedding as fallback
                agent_embedding = np.random.randn(self.embedding_config['dimensions'])
                agent_embedding = self.normalize_embedding(agent_embedding)
                agent_embeddings_map[key] = agent_embedding
                
        return agent_embeddings_map

    def initialize_df(self):
        agentKeys=["aws-claude-3-5-sonnet-v1",
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
                   "wxai-mixtral-8x7b-instruct-v01"]
        
        specializedAgentsBaseCandidates=[
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
                   "wxai-mixtral-8x7b-instruct-v01"]
        
        # Create agent embeddings first
        agent_embeddings_map = self.create_agent_embeddings(agentKeys)
        
        # Try to load the saved embeddings first
        try:
            print("Loading saved embeddings...")
            with open(embeddings_df_name, 'rb') as f:
                df = pickle.load(f)

            # Truncate embeddings here
            task_embedding_col = f'task_embedding'
            if task_embedding_col in df.columns:
                # Create embeddings for prompts by truncating the dimensions of existing embeddings
                print("Chopping existing embeddings...")
                
                # Truncate task embeddings
                df[task_embedding_col + self.embedding_config["suffix"]] = df[task_embedding_col].apply(
                    lambda x: x[:self.embedding_config["dimensions"]]
                )
                
                for key in agentKeys:
                    print(f"Creating task contexts for {key}...")
                    df[key + '_task_context'] = df.apply(
                        lambda row: self.combine_embeddings(
                            row[task_embedding_col + self.embedding_config["suffix"]],
                            agent_embeddings_map[key],
                            operation='concatenate'
                        ),
                        axis=1
                    )

                # Save the dataframe with new embeddings
                print("Saving embeddings...")
                with open(embeddings_df_name, 'wb') as f:
                    pickle.dump(df, f)
                
            # Check if we need to generate new embeddings
            task_embedding_col = f'task_embedding{self.embedding_config["suffix"]}'
            if task_embedding_col not in df.columns:
                print(f"Required embedding column {task_embedding_col} not found. Adding new embeddings...")
                
                # Create embeddings for prompts
                print("Creating new embeddings...")
                df[task_embedding_col] = [self.model.encode(
                    prompt,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=False
                ) for i, prompt in enumerate(df["prompt"])]
                
                for key in agentKeys:
                    print(f"Creating task contexts for {key}...")
                    df[key + '_task_context'] = df.apply(
                        lambda row: self.combine_embeddings(
                            row[task_embedding_col],
                            agent_embeddings_map[key],
                            operation='concatenate'
                        ),
                        axis=1
                    )

                # Save the dataframe with new embeddings
                print("Saving embeddings...")
                with open(embeddings_df_name, 'wb') as f:
                    pickle.dump(df, f)
                
        except FileNotFoundError:
            print("No saved embeddings found. Computing new embeddings...")
            df = sprout_loader.load_sprout_data()
            print("Dataset columns:", df.columns.tolist())
            
            # Create embeddings for prompts
            print("Creating embeddings...")
            task_embedding_col = f'task_embedding{self.embedding_config["suffix"]}'
            total_prompts = len(df)
            df[task_embedding_col] = [self.model.encode(
                prompt,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            ) for i, prompt in enumerate(df["prompt"])]
            
            for key in agentKeys:
                print(f"Creating task contexts for {key}...")
                df[key + '_task_context'] = df.apply(
                    lambda row: self.combine_embeddings(
                        row[task_embedding_col],
                        agent_embeddings_map[key],
                        operation='concatenate'
                    ),
                    axis=1
                )

                # Create new columns for each agent
                df[key + '_score'] = df[key].apply(lambda x: float(x['score']) if isinstance(x, dict) else None)
                df[key + '_num_input_tokens'] = df[key].apply(lambda x: int(x['num_input_tokens']) if isinstance(x, dict) else None)
                df[key + '_num_output_tokens'] = df[key].apply(lambda x: int(x['num_output_tokens']) if isinstance(x, dict) else None)
                df[key + '_judge_response'] = df[key].apply(lambda x: x.get('judge_response', None) if isinstance(x, dict) else None)
                df[key + '_response'] = df[key].apply(lambda x: x.get('response', None) if isinstance(x, dict) else None)
            
            df['time'] = range(1, len(df) + 1)
            print(df.head(2))
            print(df.describe())

            # Save the dataframe with embeddings
            print("Saving embeddings...")
            with open(embeddings_df_name, 'wb') as f:
                pickle.dump(df, f)


        # Create the final dataframe for the problem model
        print("Creating final dataframe...")
        row_list = []
        num_rounds = min(self.num_rounds, len(df))
        
        # Calculate total rows:
        # 1. Base agents: num_rounds * len(agentKeys)
        # 2. Specialized agents: For each specialized agent, it's available for (num_rounds - creation_round) rounds
        #    Number of specialized agents = num_rounds // specialized_agent_interval
        #    Each specialized agent i is created at round (i+1) * specialized_agent_interval
        #    So total specialized agent rounds = sum(num_rounds - (i+1)*specialized_agent_interval) for i in range(num_specialized_agents)
        num_specialized_agents = num_rounds // self.specialized_agent_interval
        specialized_agent_rounds = sum(
            num_rounds - (i+1)*self.specialized_agent_interval 
            for i in range(num_specialized_agents)
        )
        total_rows = (num_rounds * len(agentKeys)) + specialized_agent_rounds
        
        current_row = 0
        specialized_agents = []
        # Track (base_agent, frozenset of expert dimensions) combinations that have been used
        # Separate sets for weak and strong experts
        used_weak_specializations = set()
        used_strong_specializations = set()
        # Available expert dimensions where specializations are built
        specializable_dims = set(range((self.embedding_config['dimensions'] - 1) // 2))
        
        for time in tqdm(range(1, num_rounds + 1)):
            num_available_arms = len(agentKeys)
            # Add arms general purpose agents
            for i in range(num_available_arms):
                available_agent = agentKeys[i]
                context = df.loc[time, available_agent + '_task_context']
                base_score = df.loc[time, available_agent + '_score']
                
                # Get the agent embedding from the embeddings map
                agent_embedding = agent_embeddings_map[available_agent]
                
                true_mean = self.calculate_specialized_task_true_mean(
                    base_score,
                    context,
                    agent_embedding,
                    False
                )
                
                row_list.append((time, context, true_mean))
                current_row += 1
                if current_row % 1000 == 0:  # Log every 1000 rows to avoid too frequent updates
                    log_progress(current_row, total_rows, "Final dataframe: ")

            # Create new specialized agent
            if time % self.specialized_agent_interval == 0 and time > self.expert_start_time:
                # Try to create a new specialized agent with a unique (base_agent, dimension) combination
                max_attempts = 1  # Limit attempts to avoid infinite loops
                specialized_agent = None
                
                for _ in range(max_attempts):
                    base_agent = random.choice(specializedAgentsBaseCandidates)
                    available_dims = specializable_dims.copy()
                    expert_dims = []
                    
                    # If we have enough available dimensions, try to create a new combination
                    if len(available_dims) >= self.num_expert_dimensions:
                        # Randomly select num_expert_dimensions dimensions
                        expert_dims = sorted(random.sample(list(available_dims), self.num_expert_dimensions))
                        expert_dims_tuple = tuple(expert_dims)  # Convert to tuple for hashing
                        
                        # Determine if this will be a strong or weak expert based on time
                        is_strong_expert = time > self.expert_strength_breakpoint
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
                            # Add the combination to the appropriate set
                            used_specializations.add((base_agent, expert_dims_tuple))
                            break
                
                if specialized_agent is None:
                    print(f"Warning: Could not create new specialized agent at time {time} after {max_attempts} attempts")
            
            # Add arms for specialized agents
            for agent in specialized_agents:
                # Get the task context from the base agent's context
                task_context = df.loc[time, agent["base_agent"] + '_task_context']
                context = self.combine_embeddings(task_context[:len(task_context)//2], agent["agent_context"], "concatenate")
                
                # Calculate true mean using the helper function
                base_agent_score = df.loc[time, agent["base_agent"] + '_score']
                true_mean = self.calculate_specialized_task_true_mean(
                    base_agent_score,
                    task_context,
                    agent["agent_context"]
                )
                
                row_list.append((time, context, true_mean))
                current_row += 1
                if current_row % 1000 == 0:  # Log every 100 rows to avoid too frequent updates
                    log_progress(current_row, total_rows, "Final dataframe: ")
        
        return pd.DataFrame(row_list, columns=['time', 'context', 'true_mean'])

    def calculate_specialized_task_true_mean(self, base_agent_score, task_context, agent_context, is_specialized_agent=True):
        """
        Calculate the true mean for an agent based on:
        - 50% from base agent performance
        - 50% from task-agent fit using top num_task_fit_dimensions dimensions
        
        Args:
            base_agent_score: float, the score of the base agent
            task_context: numpy array, the task embedding (first half of concatenated context)
            agent_context: numpy array, the agent embedding
            
        Returns:
            float: The calculated true mean between 0 and 1
        """
        # Get the task embedding (first half of the concatenated context)
        task_embedding = task_context[:len(task_context)//2]
        
        # Find the dimensions with highest scalar values in task embedding
        top_dims = np.argsort(task_embedding)[-self.num_task_fit_dimensions:]
        
        # Calculate specialized component based on the top dimensions
        specialized_component = sum(task_embedding[dim] * (agent_context[dim]**100) for dim in top_dims) / self.num_task_fit_dimensions
        
        # Scale the specialized component to make sigmoid more meaningful
        scaled_component = 5.0 * specialized_component
        normalized_specialized = 1 / (1 + np.exp(-scaled_component))

        base_contribution = base_agent_score

        if (normalized_specialized < 0.6 or not is_specialized_agent):
            normalized_specialized = 0
            if(is_specialized_agent):
                base_contribution = base_contribution * 0.1
        # Combine base agent score (50%) with specialized component (50%)
        true_mean = 0.2 * base_contribution + 0.8 * normalized_specialized
        
        return true_mean