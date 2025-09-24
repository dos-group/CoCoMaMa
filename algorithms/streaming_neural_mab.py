import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from algorithms.streaming_base import StreamingAlgorithm
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


class OverallRewardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(OverallRewardNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Assuming rewards are normalized between 0 and 1
        )
    
    def forward(self, x):
        return self.network(x)


class StreamingNeuralMAB(StreamingAlgorithm):
    """
    Streaming version of the Neural MAB Paper algorithm.
    """
    
    def __init__(self, problem_model, budget, context_dim, hidden_dim: int = 64, learning_rate: float = 0.001):
        super().__init__(problem_model, budget)
        
        # Initialize neural networks
        self.input_dim = context_dim  # Context dimension from problem model
        self.single_reward_net = SingleRewardNet(self.input_dim)
        self.overall_reward_net = OverallRewardNet(budget)
        
        # Initialize optimizers
        self.single_optimizer = optim.Adam(self.single_reward_net.parameters())
        self.overall_optimizer = optim.Adam(self.overall_reward_net.parameters())
        
        # Initialize loss functions
        self.single_criterion = nn.MSELoss()
        self.overall_criterion = nn.MSELoss()
    
    def select_arms(self, available_arms: List[Arm]) -> List[Arm]:
        """Select arms using the greedy approach with neural network estimates."""
        selected_arms = []
        selected_arm_rewards = []
        remaining_arms = available_arms.copy()
        
        # Pre-compute single arm rewards for all available arms
        with torch.no_grad():
            available_arm_rewards = []
            for arm in available_arms:
                context = arm.context
                
                # Convert context to tensor if it's not already
                if not isinstance(context, torch.Tensor):
                    context_tensor = torch.FloatTensor(context)
                else:
                    context_tensor = context
                single_reward = self.single_reward_net(context_tensor).item()
                available_arm_rewards.append(single_reward)

        for _ in range(self.budget):
            if not remaining_arms:
                break
                
            best_arm = None
            best_arm_index = -1
            best_reward = float('-inf')
            
            # Try each remaining arm
            for i, arm in enumerate(remaining_arms):
                # Find the original index of this arm in available_arms
                original_index = available_arms.index(arm)
                
                # Get context features for all selected arms plus current arm
                context = selected_arm_rewards.copy()
                context.append(available_arm_rewards[original_index])
                
                # Pad context to match budget size
                while len(context) < self.budget:
                    context.append(0)
                
                # Convert context to tensor for neural network
                context_tensor = torch.FloatTensor(context)
                
                # Estimate overall reward
                with torch.no_grad():
                    estimated_reward = self.overall_reward_net(context_tensor).item()
                
                if estimated_reward > best_reward:
                    best_reward = estimated_reward
                    best_arm = arm
                    best_arm_index = i  # Index in remaining_arms
            
            if best_arm:
                selected_arms.append(best_arm)
                # Find original index to get the correct reward
                original_index = available_arms.index(best_arm)
                selected_arm_rewards.append(available_arm_rewards[original_index])
                # Remove from remaining_arms using the index in remaining_arms
                remaining_arms.pop(best_arm_index)
        
        return selected_arms
    
    def train_single_reward_net(self, contexts, rewards):
        """Train the single reward neural network."""
        contexts = torch.FloatTensor(contexts)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        
        self.single_optimizer.zero_grad()
        predictions = self.single_reward_net(contexts)
        loss = self.single_criterion(predictions, rewards)
        loss.backward()
        self.single_optimizer.step()
    
    def train_overall_reward_net(self, contexts, rewards):
        """Train the overall reward neural network."""
        contexts = torch.FloatTensor(contexts)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        
        self.overall_optimizer.zero_grad()
        predictions = self.overall_reward_net(contexts)
        loss = self.overall_criterion(predictions, rewards)
        loss.backward()
        self.overall_optimizer.step()
    
    def update(self, selected_arms: List[Arm], rewards, round_num):
        """Update the algorithm with the observed rewards."""
        # Train neural networks
        contexts = []
        for arm in selected_arms:
            contexts.append(arm.context)
        
        single_rewards = [reward.quality for reward in rewards]
        self.train_single_reward_net(contexts, single_rewards)
        
        # Prepare input for overall reward training: an ordered array of single rewards (padded to budget size)
        reward_vector = single_rewards.copy()
        while len(reward_vector) < self.budget:
            reward_vector.append(0)
        
        total_reward = sum(single_rewards)
        self.train_overall_reward_net([reward_vector], [total_reward])
