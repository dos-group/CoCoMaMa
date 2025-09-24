"""
Neural Multi-Armed Bandit (Neural-MAB) Algorithm

This implementation is based on existing paper methodology for neural network-based
multi-armed bandit algorithms. The implementation is original work.

The algorithm uses neural networks to estimate both individual arm rewards
and overall combinatorial rewards for arm selection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SingleRewardNet(nn.Module):
    """
    Neural network for estimating individual arm rewards.
    """
    def __init__(self, input_dim, hidden_dim=64):
        """
        Initialize the single reward network.
        
        Args:
            input_dim: Dimension of the input context
            hidden_dim: Dimension of the hidden layer
        """
        super(SingleRewardNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Assuming rewards are normalized between 0 and 1
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input context tensor
            
        Returns:
            torch.Tensor: Predicted reward value
        """
        return self.network(x)

class OverallRewardNet(nn.Module):
    """
    Neural network for estimating overall combinatorial rewards.
    """
    def __init__(self, input_dim, hidden_dim=64):
        """
        Initialize the overall reward network.
        
        Args:
            input_dim: Dimension of the input (budget size)
            hidden_dim: Dimension of the hidden layer
        """
        super(OverallRewardNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Assuming rewards are normalized between 0 and 1
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of selected arm rewards
            
        Returns:
            torch.Tensor: Predicted overall reward value
        """
        return self.network(x)

class NeuralMAB:
    """
    Neural Multi-Armed Bandit algorithm using neural networks for reward estimation.
    """
    def __init__(self, problem_model, budget):
        """
        Initialize the Neural MAB algorithm.
        
        Args:
            problem_model: The problem model containing arms and reward information
            budget: Number of arms to select in each round
        """
        self.problem_model = problem_model
        self.budget = budget
        self.num_rounds = problem_model.num_rounds
        
        # Get available arms to determine input dimension
        available_arms = self.problem_model.get_available_arms(1)
        if not available_arms:
            raise ValueError("No available arms found to determine input dimension")
        
        # Initialize neural networks
        self.input_dim = len(available_arms[0].context)
        self.single_reward_net = SingleRewardNet(self.input_dim)
        self.overall_reward_net = OverallRewardNet(budget)
        
        # Initialize optimizers
        self.single_optimizer = optim.Adam(self.single_reward_net.parameters())
        self.overall_optimizer = optim.Adam(self.overall_reward_net.parameters())
        
        # Initialize loss functions
        self.single_criterion = nn.MSELoss()
        self.overall_criterion = nn.MSELoss()
    
    def select_arms(self, available_arms):
        """
        Select arms using the greedy approach with neural network estimates.
        
        Args:
            available_arms: List of available arms for selection
            
        Returns:
            list: Selected arms for the current round
        """
        selected_arms = []
        selected_arm_rewards = []
        remaining_arms = available_arms.copy()
        
        # Pre-compute single arm rewards for all available arms
        with torch.no_grad():
            available_arm_rewards = []
            for arm in available_arms:
                # Convert context to tensor if it's not already
                if not isinstance(arm.context, torch.Tensor):
                    context_tensor = torch.FloatTensor(arm.context)
                else:
                    context_tensor = arm.context
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
        """
        Train the single reward neural network.
        
        Args:
            contexts: List of context vectors
            rewards: List of corresponding reward values
        """
        contexts = torch.FloatTensor(contexts)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        
        self.single_optimizer.zero_grad()
        predictions = self.single_reward_net(contexts)
        loss = self.single_criterion(predictions, rewards)
        loss.backward()
        self.single_optimizer.step()
    
    def train_overall_reward_net(self, contexts, rewards):
        """
        Train the overall reward neural network.
        
        Args:
            contexts: List of context vectors (selected arm rewards)
            rewards: List of corresponding total reward values
        """
        contexts = torch.FloatTensor(contexts)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        
        self.overall_optimizer.zero_grad()
        predictions = self.overall_reward_net(contexts)
        loss = self.overall_criterion(predictions, rewards)
        loss.backward()
        self.overall_optimizer.step()
    
    def run_algorithm(self):
        """
        Run the Neural MAB algorithm for the specified number of rounds.
        
        Returns:
            tuple: (total_reward_arr, regret_arr, played_arms_arr)
                - total_reward_arr: Array of total rewards obtained in each round
                - regret_arr: Array of regret values for each round
                - played_arms_arr: List of selected arms for each round
        """
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        played_arms_arr = []
        
        for t in range(1, self.num_rounds + 1):
            # Only log every 5% of progress
            if t % max(1, self.num_rounds // 20) == 0:
                progress = (t / self.num_rounds) * 100
                avg_reward = np.mean(total_reward_arr[:t])
                avg_regret = np.mean(regret_arr[:t])
                print(f'[Original-Neural-MAB] Progress: {progress:.1f}% | Time t = {t} | Avg Reward = {avg_reward:.3f} | Avg Regret = {avg_regret:.3f}')
            
            # Get available arms for this round
            available_arms = self.problem_model.get_available_arms(t)
            
            # Select arms using the neural network
            selected_arms = self.select_arms(available_arms)
            played_arms_arr.append(selected_arms)
            
            # Play selected arms and observe rewards
            rewards = self.problem_model.play_arms(t, selected_arms)
            
            # Calculate total reward and regret
            total_reward = self.problem_model.get_total_reward(rewards)
            regret = self.problem_model.get_regret(t, self.budget, selected_arms)
            
            total_reward_arr[t - 1] = total_reward
            regret_arr[t - 1] = regret
            
            # Train neural networks
            contexts = [arm.context for arm in selected_arms]
            single_rewards = [reward.quality for reward in rewards]
            self.train_single_reward_net(contexts, single_rewards)
            
            # Prepare context for overall reward training
            # The overall reward net expects input of size budget (list of rewards)
            overall_context = single_rewards.copy()
            # Pad to match budget size
            while len(overall_context) < self.budget:
                overall_context.append(0)
            
            self.train_overall_reward_net([overall_context], [total_reward])
        
        return total_reward_arr, regret_arr, played_arms_arr 


