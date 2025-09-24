"""
Streaming dataset implementation using Apache Arrow IPC format.
This module provides functionality to store and stream contextual bandit datasets
with task embeddings and agent embeddings/ratings.
"""

import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
from typing import List, Dict, Any, Iterator, Optional
import os


class StreamingDataset:
    """
    A streaming dataset that stores tasks with their associated agents in Arrow IPC format.
    Each task contains a task embedding and a list of available agents with their embeddings and ratings.
    """
    
    def __init__(self, task_embedding_dim: int, agent_embedding_dim: int):
        """
        Initialize the streaming dataset.
        
        Args:
            task_embedding_dim: Dimension of task embeddings
            agent_embedding_dim: Dimension of agent embeddings
        """
        self.task_embedding_dim = task_embedding_dim
        self.agent_embedding_dim = agent_embedding_dim
        
        # Define the Arrow schema
        self.schema = pa.schema([
            pa.field("task_id", pa.int32()),
            pa.field("seq", pa.int32()),  # arrival order
            pa.field("task_embedding", pa.list_(pa.float32())),
            pa.field("agents", pa.list_(pa.struct([
                pa.field("agent_id", pa.int32()),
                pa.field("agent_embedding", pa.list_(pa.float32())),
                pa.field("rating", pa.float32()),
            ]))),
        ])
    
    def write_dataset(self, tasks: List[Dict[str, Any]], output_path: str):
        """
        Write a dataset to an Arrow IPC stream file.
        
        Args:
            tasks: List of task dictionaries, each containing:
                - task_id: int
                - seq: int (arrival order)
                - task_embedding: numpy array
                - agents: list of dicts with agent_id, agent_embedding, rating
            output_path: Path to write the Arrow file
        """
        # Sort tasks by sequence number to ensure proper ordering
        tasks_sorted = sorted(tasks, key=lambda x: x['seq'])
        
        sink = pa.OSFile(output_path, "wb")
        with ipc.new_stream(sink, self.schema) as writer:
            for task in tasks_sorted:
                # Convert agents to Arrow array
                agents_data = []
                for agent in task['agents']:
                    agents_data.append({
                        'agent_id': agent['agent_id'],
                        'agent_embedding': agent['agent_embedding'].tolist(),
                        'rating': agent['rating']
                    })
                
                agents_array = pa.array([agents_data], type=self.schema.field("agents").type)
                
                # Create record batch
                batch = pa.record_batch([
                    pa.array([task['task_id']], pa.int32()),
                    pa.array([task['seq']], pa.int32()),
                    pa.array([task['task_embedding'].tolist()], pa.list_(pa.float32())),
                    agents_array
                ], schema=self.schema)
                
                writer.write(batch)
    
    def stream_tasks(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Stream tasks from an Arrow IPC file one at a time.
        
        Args:
            file_path: Path to the Arrow file
            
        Yields:
            Dictionary containing task data for each task
        """
        with pa.memory_map(file_path, "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                # Convert batch to Python objects
                for row in batch.to_pylist():
                    # Convert embeddings back to numpy arrays
                    task_data = {
                        'task_id': row['task_id'],
                        'seq': row['seq'],
                        'task_embedding': np.array(row['task_embedding'], dtype=np.float32),
                        'agents': []
                    }
                    
                    # Convert agent data
                    for agent in row['agents']:
                        task_data['agents'].append({
                            'agent_id': agent['agent_id'],
                            'agent_embedding': np.array(agent['agent_embedding'], dtype=np.float32),
                            'rating': agent['rating']
                        })
                    
                    yield task_data
    
    def get_dataset_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about the dataset without loading it entirely.
        
        Args:
            file_path: Path to the Arrow file
            
        Returns:
            Dictionary with dataset information
        """
        if not os.path.exists(file_path):
            return {'error': 'File does not exist'}
        
        file_size = os.path.getsize(file_path)
        
        # Read just the first batch to get schema info
        with pa.memory_map(file_path, "r") as source:
            reader = ipc.open_stream(source)
            first_batch = next(reader)
            
            return {
                'file_size_mb': file_size / (1024 * 1024),
                'schema': str(self.schema),
                'task_embedding_dim': self.task_embedding_dim,
                'agent_embedding_dim': self.agent_embedding_dim,
                'first_task_sample': first_batch.to_pylist()[0] if len(first_batch) > 0 else None
            }


class StreamingProblemModel:
    """
    A problem model that works with streaming datasets.
    This replaces the traditional problem model interface to work with streaming data.
    """
    
    def __init__(self, dataset_path: str, task_embedding_dim: int, agent_embedding_dim: int, 
                 num_rounds: int, budget: int):
        """
        Initialize the streaming problem model.
        
        Args:
            dataset_path: Path to the Arrow dataset file
            task_embedding_dim: Dimension of task embeddings
            agent_embedding_dim: Dimension of agent embeddings
            num_rounds: Number of rounds to run
            budget: Budget (number of agents to select per round)
        """
        self.dataset_path = dataset_path
        self.task_embedding_dim = task_embedding_dim
        self.agent_embedding_dim = agent_embedding_dim
        self.num_rounds = num_rounds
        self.budget = budget
        
        # Don't create the iterator here - it will be created when needed
        # This makes the object pickleable for parallel processing
        self.current_task = None
        self.current_round = 0
        self.task_iterator = None
        self._initialized = False
    
    def _load_next_task(self):
        """Load the next task from the stream."""
        # Create iterator if it doesn't exist
        if self.task_iterator is None:
            streaming_dataset = StreamingDataset(self.task_embedding_dim, self.agent_embedding_dim)
            self.task_iterator = streaming_dataset.stream_tasks(self.dataset_path)
        
        try:
            self.current_task = next(self.task_iterator)
            if not self._initialized:
                self._initialized = True
                self.current_round = 1  # First task is round 1
            else:
                self.current_round += 1
        except StopIteration:
            self.current_task = None
    
    def get_available_arms(self, t: int) -> List['Arm']:
        """
        Get available arms (agents) for the current task.
        
        Args:
            t: Time step (round number)
            
        Returns:
            List of Arm objects
        """
        # Ensure we have a task loaded
        if self.current_task is None:
            self._load_next_task()
        
        if self.current_task is None or self.current_round != t:
            raise ValueError(f"No task available for round {t}")
        
        arm_list = []
        for i, agent in enumerate(self.current_task['agents']):
            # Create context by concatenating task and agent embeddings
            context = np.concatenate([
                self.current_task['task_embedding'],
                agent['agent_embedding']
            ])
            arm_list.append(Arm(i, context, agent['rating']))
        
        return arm_list
    
    def get_regret(self, t: int, budget: int, slate: List['Arm']) -> float:
        """
        Calculate regret for the selected slate.
        
        Args:
            t: Time step
            budget: Budget (unused, kept for compatibility)
            slate: Selected arms
            
        Returns:
            Regret value
        """
        # Ensure we have a task loaded
        if self.current_task is None:
            self._load_next_task()
        
        if self.current_task is None or self.current_round != t:
            raise ValueError(f"No task available for round {t}")
        
        # Get all possible rewards
        all_rewards = [Reward(Arm(i, 
                                 np.concatenate([self.current_task['task_embedding'], 
                                               agent['agent_embedding']]), 
                                 agent['rating']), 
                             agent['rating']) 
                      for i, agent in enumerate(self.current_task['agents'])]
        
        # Get optimal reward
        optimal_reward = self.get_total_reward(all_rewards)
        
        # Get selected rewards
        selected_rewards = []
        for item in slate:
            # Handle both arm objects and agent dictionaries
            if hasattr(item, 'true_mean'):
                # It's an arm object
                arm = item
                selected_rewards.append(Reward(arm, arm.true_mean))
            elif isinstance(item, dict) and 'arm' in item:
                # It's an agent dictionary
                arm = item['arm']
                selected_rewards.append(Reward(arm, arm.true_mean))
            else:
                raise ValueError(f"Unexpected item type in slate: {type(item)}")
        
        selected_reward = self.get_total_reward(selected_rewards)
        
        return optimal_reward - selected_reward
    
    def get_total_reward(self, rewards: List['Reward']) -> float:
        """
        Calculate total reward from a list of rewards.
        Uses max reward strategy (can be modified for other strategies).
        
        Args:
            rewards: List of Reward objects
            
        Returns:
            Total reward value
        """
        if not rewards:
            return 0.0
        return max(reward.quality for reward in rewards)
    
    def play_arms(self, t: int, slate: List['Arm']) -> List['Reward']:
        """
        Play the selected arms and return rewards.
        
        Args:
            t: Time step
            slate: Selected arms
            
        Returns:
            List of Reward objects
        """
        # Ensure we have a task loaded
        if self.current_task is None:
            self._load_next_task()
        
        if self.current_task is None or self.current_round != t:
            raise ValueError(f"No task available for round {t}")
        
        reward_list = []
        for item in slate:
            # Handle both arm objects and agent dictionaries
            if hasattr(item, 'true_mean'):
                # It's an arm object
                arm = item
                reward_list.append(Reward(arm, arm.true_mean))
            elif isinstance(item, dict) and 'arm' in item:
                # It's an agent dictionary
                arm = item['arm']
                reward_list.append(Reward(arm, arm.true_mean))
            else:
                raise ValueError(f"Unexpected item type in slate: {type(item)}")
        
        return reward_list
    
    def advance_to_next_task(self):
        """Manually advance to the next task. Call this after processing a task."""
        self._load_next_task()
    
    def oracle(self, K: int, g_list: List[float]) -> List[int]:
        """
        Oracle selection of top K arms based on their true means.
        
        Args:
            K: Number of arms to select
            g_list: List of arm qualities
            
        Returns:
            List of indices of selected arms
        """
        g_list = np.array(g_list)
        # Add small random tie-breaker
        noise = np.random.rand(len(g_list)) * 1e-8
        randomized_g_list = g_list + noise
        return np.argsort(randomized_g_list)[-K:].tolist()
    
    def get_size(self) -> int:
        """
        Get the total size of the dataset.
        Note: This requires reading the entire file, so it's expensive.
        
        Returns:
            Total number of arms across all tasks
        """
        total_size = 0
        streaming_dataset = StreamingDataset(self.task_embedding_dim, self.agent_embedding_dim)
        for task in streaming_dataset.stream_tasks(self.dataset_path):
            total_size += len(task['agents'])
        return total_size


# Import Arm and Reward classes for compatibility
from Arm import Arm
from Reward import Reward
