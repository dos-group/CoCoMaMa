import numpy as np
from typing import List
from algorithms.streaming_base import StreamingAlgorithm
from algorithms import random_algo
from Arm import Arm


class StreamingCCMAB(StreamingAlgorithm):
    """
    Streaming version of the CC-MAB algorithm.
    """
    
    def __init__(self, problem_model, budget, context_dim):
        super().__init__(problem_model, budget)
        self.context_dim = context_dim
        
        # Initialize context normalization parameters
        self.context_min = None
        self.context_max = None
        self.context_range = None
        
        # Initialize hypercube tracking
        self.hypercube_played_counter_dict = {}
        self.avg_reward_dict = {}  # maps hypercube to avg reward
    
    def normalize_context(self, context):
        """
        Normalize context to [0,1] range.
        If normalization parameters haven't been set yet, estimate them from the current context.
        """
        if self.context_min is None:
            # First time seeing a context, initialize normalization parameters
            # We'll use a conservative estimate based on the current context
            # and update as we see more contexts
            self.context_min = np.min(context) - 0.1  # Add small buffer
            self.context_max = np.max(context) + 0.1  # Add small buffer
            self.context_range = self.context_max - self.context_min
            
            # Ensure we don't have zero range
            if self.context_range < 1e-10:
                self.context_range = 1.0
        
        # Normalize to [0,1]
        normalized = (context - self.context_min) / self.context_range
        
        # Clip to ensure we stay in [0,1] range
        normalized = np.clip(normalized, 0, 1)
        
        return normalized

    def get_hypercube_of_context(self, context, round_num):
        """Get the hypercube that contains the given context."""
        # Normalize the context first
        normalized_context = self.normalize_context(context)
        
        # TODO: Implementation differs from the original CC-MAB paper
        # Calculate hT based on current round
        hT = np.ceil(round_num ** (1 / (3 + self.context_dim)))
        cube_length = 1 / hT
        
        return tuple((normalized_context / cube_length).astype(int))
    
    def select_arms(self, available_arms: List[Arm]) -> List[Arm]:
        """Select arms using CC-MAB algorithm."""
        round_num = len(self.hypercube_played_counter_dict) + 1
        
        # Group arms by their hypercubes
        arrived_cube_arms_dict = {}
        arrived_cube_set = set()
        
        for available_arm in available_arms:
            context = available_arm.context
            hypercube = self.get_hypercube_of_context(context, round_num)
            if hypercube not in arrived_cube_arms_dict:
                arrived_cube_arms_dict[hypercube] = list()
            arrived_cube_arms_dict[hypercube].append(available_arm)
            arrived_cube_set.add(hypercube)

        # Identify underexplored hypercubes
        underexplored_arms = []
        for cube in arrived_cube_set:
            if self.hypercube_played_counter_dict.get(cube, 0) <= round_num ** (2 / (3 + self.context_dim)) * np.log(round_num):
                underexplored_arms.extend(arrived_cube_arms_dict[cube])

        # Play arms
        if len(underexplored_arms) >= self.budget:
            slate = random_algo.sample(underexplored_arms, self.budget)
        else:
            slate = []
            slate.extend(underexplored_arms)
            # Get arms that are not in underexplored_arms
            underexplored_arm_ids = [id(arm) for arm in underexplored_arms]
            not_chosen_arms = [arm for arm in available_arms if id(arm) not in underexplored_arm_ids]
            
            if len(not_chosen_arms) > 0:
                conf_list = np.empty(len(not_chosen_arms))
                for i, arm in enumerate(not_chosen_arms):
                    context = arm.context
                    conf_list[i] = self.avg_reward_dict.get(self.get_hypercube_of_context(context, round_num), 0)
                
                arm_indices = self.problem_model.oracle(self.budget - len(slate), conf_list)
                for index in arm_indices:
                    selected_arm = not_chosen_arms[index]
                    slate.append(selected_arm)
        
        return slate
    
    def update(self, selected_arms: List[Arm], reward_values, round_num):
        """Update the algorithm with the observed rewards."""
        rewards = reward_values
        
        # Update the counters
        for reward in rewards:
            cube_with_context = self.get_hypercube_of_context(reward.context, round_num)
            new_counter = self.hypercube_played_counter_dict[cube_with_context] = self.hypercube_played_counter_dict.get(
                cube_with_context, 0) + 1
            self.avg_reward_dict[cube_with_context] = (self.avg_reward_dict.get(cube_with_context, 0) * (
                    new_counter - 1) + reward.quality) / new_counter
