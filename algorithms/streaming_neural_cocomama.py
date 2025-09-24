import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple
from algorithms.streaming_cocoma import StreamingCoCoMaMa
from UcbNode import UcbNode
from Hyperrectangle import Hyperrectangle
from Reward import Reward
from Arm import Arm

class SingleRewardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SingleRewardNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Assuming rewards are normalized between 0 and 1
        )
    
    def forward(self, x):
        return self.network(x)


class StreamingNeuralCoCoMaMa(StreamingCoCoMaMa):
    """
    Streaming version of Neural CoCoMaMa algorithm.
    Combines neural network with CoCoMaMa-style tree structure.
    """
    
    def __init__(self, problem_model, v1: float, v2: float, N: int, rho: float,
                 budget: int, initial_hypercube: Hyperrectangle, context_dim: int, 
                 hidden_dim: int = 64, learning_rate: float = 0.001):
        """
        Initialize streaming Neural CoCoMaMa.
        
        Args:
            problem_model: Streaming problem model
            v1: Algorithm parameter
            v2: Algorithm parameter
            N: Algorithm parameter
            rho: Algorithm parameter
            budget: Budget
            initial_hypercube: Initial hypercube
            context_dim: Dimension of context vectors
            hidden_dim: Hidden dimension of neural network
            learning_rate: Learning rate for optimizer
        """
        super().__init__(problem_model, v1, v2, N, rho, budget, initial_hypercube)


               
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.single_reward_net = SingleRewardNet(self.context_dim, self.hidden_dim)
        # Initialize optimizers
        self.single_optimizer = optim.Adam(self.single_reward_net.parameters())
        # Initialize loss functions
        self.single_criterion = nn.MSELoss()

    
    def select_arms(self, available_arms: List[Arm]) -> List[Arm]:
        """Select arms using Neural CoCoMaMa algorithm."""# Select top K arms
        if len(available_arms) <= self.budget:
            return available_arms
        
        # Calculate index values for each arm
        arm_indices = []
        for arm in available_arms:
            # Find the node containing the arm context
            context = arm.context
            node = self._find_node_containing_context(context, self.leaves)
            
            # Calculate index value
            arm_index = self._get_arm_index(node, len(self.leaves_count_arr) + 1, context)
            arm_indices.append(arm_index)
        
        # Get indices of top K arms
        top_indices = np.argsort(arm_indices)[-self.budget:]
        return [available_arms[i] for i in top_indices]
    
    def update(self, selected_arms: List[Arm], 
               rewards: List[Reward], round_num: int):
        """Update both neural network and tree structure."""
        contexts = [reward.context for reward in rewards]
        single_rewards = [reward.quality for reward in rewards]
        self.train_single_reward_net(contexts, single_rewards)
        super().update(selected_arms, rewards, round_num)

    def train_single_reward_net(self, contexts: list, rewards: list) -> None:
        """
        Train the single reward neural network on the provided contexts and rewards.

        Args:
            contexts (list): List of context vectors (each a list or np.ndarray of floats).
            rewards (list): List of reward values (floats) corresponding to each context.
        """
        contexts_tensor = torch.FloatTensor(contexts)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        
        self.single_optimizer.zero_grad()
        predictions = self.single_reward_net(contexts_tensor)
        loss = self.single_criterion(predictions, rewards_tensor)
        loss.backward()
        self.single_optimizer.step()

    
    def _get_arm_index(self, node: 'UcbNode', t: int, context: np.ndarray) -> float:
        """
        Compute the index value for an arm using the neural network's reward estimate and confidence bound.

        Args:
            node (UcbNode): The node containing the arm's context.
            t (int): The current round number.
            context (np.ndarray): The context vector for the arm.

        Returns:
            float: The computed index value for the arm.
        """
        node_key = id(node)
        num_times_node_played = self.node_played_counter_dict.get(node_key, 0)
        num_times_parent_node_played = self.node_played_counter_dict.get(id(node.parent_node), 0)
        combined_confidence = self._calc_confidence(num_times_node_played + num_times_parent_node_played, t)
        # Estimate overall reward using the neural network
        with torch.no_grad():
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
            estimated_reward = self.single_reward_net(context_tensor).item()
            return estimated_reward + combined_confidence
