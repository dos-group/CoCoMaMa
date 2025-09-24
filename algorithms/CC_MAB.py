import numpy as np

import ProblemModel
from algorithms import random_algo


def find_node_containing_context(context, leaves):
    """
    Find the node that contains the given context.
    
    Args:
        context: The context vector to find a node for
        leaves: List of leaf nodes in the tree
        
    Returns:
        UcbNode: The node containing the context, or None if not found
    """
    for leaf in leaves:
        if leaf.contains_context(context):
            return leaf


"""
Contextual Combinatorial Multi-Armed Bandit (CC-MAB) Algorithm

This implementation is based on the work by Chen, Lixing, Jie Xu, and Zhuo Lu. "Contextual combinatorial multi-armed bandits with volatile arms and submodular reward." Advances in Neural Information Processing Systems 31 (2018).

Implementation also available at: https://github.com/Bilkent-CYBORG/ACC-UCB by Nika, Andi, Sepehr Elahi, and Cem Tekin.

The algorithm handles contextual combinatorial multi-armed bandit problems
using hypercube-based discretization of the context space.
"""


class CCMAB:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, budget, context_dim):
        """
        Initialize the CC-MAB algorithm.
        
        Args:
            problem_model: The problem model containing arms and reward information
            budget: Number of arms to select in each round
            context_dim: Dimension of the context space
        """
        self.context_dim = context_dim
        self.num_rounds = problem_model.num_rounds
        self.hT = np.ceil(self.num_rounds ** (1 / (3 + context_dim)))
        self.cube_length = 1 / self.hT
        self.budget = budget
        self.problem_model = problem_model
        
        # Initialize context normalization parameters
        self.context_min = None
        self.context_max = None
        self.context_range = None

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

    def get_hypercube_of_context(self, context):
        """
        Get the hypercube index for a given context.
        
        Args:
            context: The context vector
            
        Returns:
            tuple: The hypercube coordinates as a tuple of integers
        """
        # Normalize the context first
        normalized_context = self.normalize_context(context)
        return tuple((normalized_context / self.cube_length).astype(int))

    def run_algorithm(self):
        """
        Run the CC-MAB algorithm for the specified number of rounds.
        
        Returns:
            tuple: (total_reward_arr, regret_arr)
                - total_reward_arr: Array of total rewards obtained in each round
                - regret_arr: Array of regret values for each round
        """
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        hypercube_played_counter_dict = {}
        avg_reward_dict = {}  # maps hypercube to avg reward

        for t in range(1, self.num_rounds + 1):
            arrived_cube_arms_dict = {}
            available_arms = self.problem_model.get_available_arms(t)

            # Hypercubes that the arrived arms belong to
            arrived_cube_set = set()
            for available_arm in available_arms:
                hypercube = self.get_hypercube_of_context(available_arm.context)
                if hypercube not in arrived_cube_arms_dict:
                    arrived_cube_arms_dict[hypercube] = list()
                arrived_cube_arms_dict[hypercube].append(available_arm)
                arrived_cube_set.add(hypercube)

            # Identify underexplored hypercubes
            underexplored_arm_set = set()
            for cube in arrived_cube_set:
                if hypercube_played_counter_dict.get(cube, 0) <= t ** (2 / (3 + self.context_dim)) * np.log(t):
                    underexplored_arm_set.update(arrived_cube_arms_dict[cube])

            # Play arms
            if len(underexplored_arm_set) >= self.budget:
                slate = random_algo.sample(list(underexplored_arm_set), self.budget)
            else:
                slate = []
                slate.extend(underexplored_arm_set)
                not_chosen_arms = list(set(available_arms) - underexplored_arm_set)
                i = 0
                conf_list = np.empty(len(not_chosen_arms))
                for arm in not_chosen_arms:
                    conf_list[i] = avg_reward_dict.get(self.get_hypercube_of_context(arm.context), 0)
                    i += 1
                arm_indices = self.problem_model.oracle(self.budget - len(slate), conf_list)
                for index in arm_indices:
                    selected_arm = not_chosen_arms[index]
                    slate.append(selected_arm)

            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards)
            regret_arr[t - 1] = self.problem_model.get_regret(t, self.budget, slate)

            # Update the counters
            for reward in rewards:
                cube_with_context = self.get_hypercube_of_context(reward.context)
                new_counter = hypercube_played_counter_dict[cube_with_context] = hypercube_played_counter_dict.get(
                    cube_with_context, 0) + 1
                avg_reward_dict[cube_with_context] = (avg_reward_dict.get(cube_with_context, 0) * (
                        new_counter - 1) + reward.quality) / new_counter

        return total_reward_arr, regret_arr


