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

saved_df_name = 'sprout_simulation_df.pkl'  # file where the saved simulation-ready dataframe will be saved
embeddings_df_name = 'sprout_embeddings_df.pkl'  # file where the embeddings will be saved

# Set up cache directory in the virtual environment
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cache')
os.makedirs(cache_dir, exist_ok=True)

def log_progress(current, total, prefix=""):
    """Log progress for every 1% completed"""
    percentage = (current / total) * 100
    # Log every time we've completed another 1% of total, regardless of whether total is a multiple of 100
    if total > 0 and current % max(1, total // 100) == 0:
        print(f"{prefix}Progress: {percentage:.1f}% ({current}/{total})")

class SPROUTProblemModel(ProblemModel):
    def __init__(self, num_rounds, budget, use_saved, embedding_config=None):
        self.num_rounds = num_rounds
        self.embedding_config = embedding_config or {
            'model_name': 'all-MiniLM-L6-v2',
            'dimensions': 384,
            'suffix': ''
        }
        
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
                
                # Initialize the sentence transformer model for computing agent embeddings
                print(f"Loading embedding model: {self.embedding_config['model_name']}...")
                model = SentenceTransformer(
                    self.embedding_config['model_name'],
                    device='cpu',
                    trust_remote_code=True,
                    truncate_dim=self.embedding_config['dimensions'],
                    cache_folder=cache_dir
                )
                
                # Truncate task embeddings
                df[task_embedding_col + self.embedding_config["suffix"]] = df[task_embedding_col].apply(
                    lambda x: x[:self.embedding_config["dimensions"]]
                )
                
                for key in agentKeys:
                    print(f"Creating task contexts for {key}...")

                    # Load the model card JSON file for this agent
                    with open(f'model_cards/{key}.json', 'r') as f:
                        agent_card = json.load(f)
                        # Convert JSON to string before encoding
                        agent_card_str = json.dumps(agent_card)
                        # Compute and truncate agent embeddings
                        agent_embedding = model.encode(
                            agent_card_str,
                            convert_to_tensor=False,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )
                        df[key + '_task_context'] = df.apply(
                            lambda row: self.combine_embeddings(
                                row[task_embedding_col + self.embedding_config["suffix"]],
                                agent_embedding,
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
                # Initialize the sentence transformer model
                print(f"Loading embedding model: {self.embedding_config['model_name']}...")
                model = SentenceTransformer(
                    self.embedding_config['model_name'],
                    device='cpu',
                    trust_remote_code=True,
                    truncate_dim=self.embedding_config['dimensions'],
                    cache_folder=cache_dir
                )
                
                # Create embeddings for prompts
                print("Creating new embeddings...")
                df[task_embedding_col] = df["prompt"].apply(
                    lambda x, idx=df.index.get_loc(df["prompt"].name): (
                        log_progress(idx + 1, total_prompts, "Prompts: "),
                        model.encode(
                            x,
                            convert_to_tensor=False,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )
                    )[1]
                )
                
                for key in agentKeys:
                    print(f"Creating task contexts for {key}...")

                    # Load the model card JSON file for this agent
                    with open(f'model_cards/{key}.json', 'r') as f:
                        agent_card = json.load(f)
                        # Convert JSON to string before encoding
                        agent_card_str = json.dumps(agent_card)
                        agent_embedding = model.encode(
                            agent_card_str,
                            convert_to_tensor=False,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )
                        df[key + '_task_context'] = df.apply(
                            lambda row: self.combine_embeddings(
                                row[task_embedding_col],
                                agent_embedding,
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
            
            # Initialize the sentence transformer model
            print(f"Loading embedding model: {self.embedding_config['model_name']}...")
            model = SentenceTransformer(
                self.embedding_config['model_name'],
                device='cpu',
                trust_remote_code=True,
                truncate_dim=self.embedding_config['dimensions'],
                cache_folder=cache_dir
            )
            
            # Create embeddings for prompts
            print("Creating embeddings...")
            task_embedding_col = f'task_embedding{self.embedding_config["suffix"]}'
            total_prompts = len(df)
            df[task_embedding_col] = df["prompt"].apply(
                lambda x, idx=df.index.get_loc(df["prompt"].name): (
                    log_progress(idx + 1, total_prompts, "Prompts: "),
                    model.encode(
                        x,
                        convert_to_tensor=False,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                )[1]
            )
            
            for key in agentKeys:
                print("Creating task contexts...")

                # Load the model card JSON file for this agent
                with open(f'model_cards/{key}.json', 'r') as f:
                    agent_card = json.load(f)
                    # Convert JSON to string before encoding
                    agent_card_str = json.dumps(agent_card)
                    agent_embedding = model.encode(
                        agent_card_str,
                        convert_to_tensor=False,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    df[key + '_task_context'] = df.apply(
                        lambda row: self.combine_embeddings(
                            row[task_embedding_col],
                            agent_embedding,
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
        total_rows = num_rounds * len(agentKeys)
        current_row = 0
        
        for time in tqdm(range(1, num_rounds + 1)):
            num_available_arms = len(agentKeys)
            for i in range(num_available_arms):
                available_agent = agentKeys[i]
                context = df.loc[time, available_agent + '_task_context']
                true_mean = df.loc[time, available_agent + '_score']
                row_list.append((time, context, true_mean))
                current_row += 1
                log_progress(current_row, total_rows, "Final dataframe: ")
                
        return pd.DataFrame(row_list, columns=['time', 'context', 'true_mean'])