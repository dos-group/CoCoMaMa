"""
Base class for streaming contextual bandit algorithms.
This provides a common interface for algorithms that work with streaming datasets.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from streaming_dataset import StreamingProblemModel
from Reward import Reward
from Arm import Arm


class StreamingAlgorithm(ABC):
    """
    Abstract base class for streaming contextual bandit algorithms.
    """
    
    def __init__(self, problem_model: StreamingProblemModel, budget: int):
        """
        Initialize the streaming algorithm.
        
        Args:
            problem_model: Streaming problem model
            budget: Budget (number of agents to select per round)
        """
        self.problem_model = problem_model
        self.budget = budget
        self.num_rounds = problem_model.num_rounds
    
    @abstractmethod
    def select_arms(
        self, 
        available_arms: List['Arm']
    ) -> List['Arm']:
        """
        Select arms for a given task.
        
        Args:
            available_arms: List of available Arm objects
            
        Returns:
            List of selected Arm objects
        """
        pass
    
    @abstractmethod
    def update(self, selected_arms: List['Arm'], 
               rewards: List[Reward], round_num: int):
        """
        Update the algorithm with observed rewards.
        
        Args:
            selected_arms: List of selected Arm objects
            rewards: Observed rewards for selected arms
            round_num: Current round number
        """
        pass
    
    def run_algorithm(self) -> Tuple[np.ndarray, np.ndarray, List[List[Any]]]:
        """
        Run the streaming algorithm.
        
        Returns:
            Tuple of (rewards, regrets, played_arms)
        """
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        played_arms_arr = []
        
        for t in range(1, self.num_rounds + 1):
            # Ensure we're at the correct round
            if t == 1 and self.problem_model.current_task is None:
                # Load the first task
                self.problem_model._load_next_task()
            
            # Get available arms for current task
            available_arms = self.problem_model.get_available_arms(t)
            
            if not available_arms:
                # No arms available, skip this round
                total_reward_arr[t - 1] = 0
                regret_arr[t - 1] = 0
                played_arms_arr.append([])
                continue
            
            # Select arms directly
            selected_arms = self.select_arms(available_arms)
            
            # Play arms and get rewards
            rewards = self.problem_model.play_arms(t, selected_arms)
            reward_values = [reward.quality for reward in rewards]
            
            # Calculate regret
            regret_arr[t - 1] = self.problem_model.get_regret(t, self.budget, selected_arms)
            
            # Store results
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards)
            played_arms_arr.append(selected_arms)
            
            # Update algorithm
            self.update(selected_arms, rewards, t)
            
            # Advance to next task
            self.problem_model.advance_to_next_task()
        
        return total_reward_arr, regret_arr, played_arms_arr


class StreamingRandom(StreamingAlgorithm):
    """
    Random selection algorithm for streaming datasets.
    """
    
    def __init__(self, problem_model: StreamingProblemModel, budget: int):
        super().__init__(problem_model, budget)
    
    def select_arms(self, available_arms: List[Arm]) -> List[Arm]:
        """Randomly select arms."""
        num_arms = len(available_arms)
        if num_arms <= self.budget:
            return available_arms
        
        selected_indices = np.random.choice(num_arms, size=self.budget, replace=False).tolist()
        return [available_arms[i] for i in selected_indices]
    
    def update(self, selected_arms: List[Arm], 
               rewards: List[Reward], round_num: int):
        """No update needed for random algorithm."""
        pass


class StreamingBenchmark(StreamingAlgorithm):
    """
    Oracle benchmark algorithm for streaming datasets.
    """
    
    def __init__(self, problem_model: StreamingProblemModel, budget: int):
        super().__init__(problem_model, budget)
        self.uniquely_best_arms_arr = []
    
    def select_arms(self, available_arms: List[Arm]) -> List[Arm]:
        """Select arms with highest true_mean values."""
        # Sort by true_mean
        sorted_arms = sorted(available_arms, key=lambda x: x.true_mean, reverse=True)
        
        # Find highest true_mean
        highest_true_mean = sorted_arms[0].true_mean
        
        # Count arms with highest true_mean
        num_best = sum(1 for arm in sorted_arms if arm.true_mean == highest_true_mean)
        
        # Store uniquely best arms info
        if num_best == 1:
            self.uniquely_best_arms_arr.append([sorted_arms[0]])
        else:
            self.uniquely_best_arms_arr.append([])
        
        # Select all arms with highest true_mean
        best_arms = []
        for arm in available_arms:
            if arm.true_mean == highest_true_mean:
                best_arms.append(arm)
        
        return best_arms
    
    def update(self, selected_arms: List[Arm], 
               rewards: List[Reward], round_num: int):
        """No update needed for oracle algorithm."""
        pass
    
    def run_algorithm(self) -> Tuple[np.ndarray, np.ndarray, List[List[Any]], List[List[Any]]]:
        """
        Run the benchmark algorithm.
        
        Returns:
            Tuple of (rewards, regrets, played_arms, uniquely_best_arms)
        """
        rewards, regrets, played_arms = super().run_algorithm()
        return rewards, regrets, played_arms, self.uniquely_best_arms_arr
