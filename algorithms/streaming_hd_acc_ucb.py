"""
Streaming High-Dimensional Adaptive Contextual Combinatorial Upper Confidence Bound (Streaming HD-ACC-UCB) Algorithm

This is a modified version of the ACC-UCB algorithm adapted for streaming data scenarios.
The HD-ACC-UCB variant uses Hyperrectangles instead of Hypercubes for high-dimensional contexts.
Based on the original work by A. Nika, S. Elahi and C. Tekin:
"Contextual combinatorial volatile multi-armed bandit with adaptive discretization"
23rd International Conference on Artificial Intelligence and Statistics (AISTATS), 2024.

Original implementation available at: https://github.com/Bilkent-CYBORG/ACC-UCB

The streaming version handles data that arrives sequentially and may not fit in memory.
"""

import math
from math import sqrt
import numpy as np
from typing import List
from algorithms.streaming_base import StreamingAlgorithm
from UcbNode import UcbNode
from algorithms import random_algo
from Arm import Arm


def find_node_containing_context(context, leaves):
    """
    Find the node that contains the given context or the closest node if none contains it.
    
    Args:
        context: The context vector to find a node for
        leaves: List of leaf nodes in the tree
        
    Returns:
        UcbNode: The node containing the context or the closest node
    """
    # First try to find a node that contains the context
    for leaf in leaves:
        if leaf.contains_context(context):
            return leaf
    
    # If no node contains the context, find the closest node
    min_distance = float('inf')
    closest_node = None
    for leaf in leaves:
        # Calculate distance to center of hyperrectangle
        distance = np.linalg.norm(context - leaf.hyperrectangle.center)
        if distance < min_distance:
            min_distance = distance
            closest_node = leaf
    
    return closest_node


class StreamingHDACCUCB(StreamingAlgorithm):
    """
    Streaming version of the HD-ACC-UCB algorithm for handling sequential data.
    This variant uses Hyperrectangles instead of Hypercubes for high-dimensional contexts.
    """
    
    def __init__(self, problem_model, v1, v2, N, rho, budget, initial_hyperrectangle):
        """
        Initialize the Streaming HD-ACC-UCB algorithm.
        
        Args:
            problem_model: The streaming problem model containing arms and reward information
            v1: First parameter for the confidence bound calculation
            v2: Second parameter for the confidence bound calculation
            N: Number of dimensions for the binary tree (must be 2)
            rho: Parameter controlling the splitting threshold
            budget: Number of arms to select in each round
            initial_hyperrectangle: Initial hyperrectangle defining the context space
        """
        super().__init__(problem_model, budget)
        self.initial_hyperrectangle = initial_hyperrectangle
        if N != 2:
            print('HD-ACC-UCB ONLY works when N = 2')
            exit(1)
        self.N = N
        self.rho = rho
        self.v2 = v2
        self.v1 = v1
        
        # Initialize the tree structure
        self.leaves = [UcbNode(None, 0, self.initial_hyperrectangle)]
        self.node_played_counter_dict = {}
        self.avg_reward_dict = {}
        self.leaves_count_arr = []
    
    def select_arms(self, available_arms: List[Arm]) -> List[Arm]:
        """Select arms using HD-ACC-UCB algorithm."""
        index_list = np.zeros(len(available_arms))
        round_num = len(self.leaves_count_arr) + 1
        
        # Check if only root node is available
        if len(self.leaves) == 1:
            arm_indices_to_play = random_algo.sample(range(len(available_arms)), self.budget)
        else:
            for i, available_arm in enumerate(available_arms):
                context = available_arm.context
                node = find_node_containing_context(context, self.leaves)
                index_list[i] = self.get_arm_index(node, self.node_played_counter_dict, self.avg_reward_dict, round_num)
            
            arm_indices_to_play = self.problem_model.oracle(self.budget, index_list)
        
        selected_arms = [available_arms[index] for index in arm_indices_to_play]
        return selected_arms
    
    def update(self, selected_arms: List[Arm], reward_values, round_num):
        """Update the algorithm with the observed rewards."""
        rewards = reward_values
        
        # Find nodes for selected arms
        selected_nodes = set()
        for selected_arm in selected_arms:
            context = selected_arm.context
            node = find_node_containing_context(context, self.leaves)
            selected_nodes.add(node)
            # Initialize the node in the counter dictionary if it doesn't exist
            if node not in self.node_played_counter_dict:
                self.node_played_counter_dict[node] = 0
            if node not in self.avg_reward_dict:
                self.avg_reward_dict[node] = 0
        
        # Update the counters
        for reward in rewards:
            node_with_context = find_node_containing_context(reward.context, selected_nodes)
            new_counter = self.node_played_counter_dict[node_with_context] = self.node_played_counter_dict.get(
                node_with_context, 0) + 1
            self.avg_reward_dict[node_with_context] = (self.avg_reward_dict.get(node_with_context, 0) * (
                        new_counter - 1) + reward.quality) / new_counter

        for selected_node in selected_nodes:
            # Ensure the node exists in the counter dictionary before accessing it
            if selected_node not in self.node_played_counter_dict:
                self.node_played_counter_dict[selected_node] = 0
            if selected_node not in self.avg_reward_dict:
                self.avg_reward_dict[selected_node] = 0
                
            # Split the node if needed
            if self.calc_confidence(
                    self.node_played_counter_dict[selected_node], round_num) <= self.v1 * self.rho ** selected_node.h:
                produced_nodes = selected_node.reproduce()
                self.leaves.remove(selected_node)
                self.leaves.extend(produced_nodes)
        
        # Record leaves count
        self.leaves_count_arr.append(len(self.leaves))
    
    def get_arm_index(self, node, node_played_counter_dict, avg_reward_dict, t):
        """Calculate the UCB index for an arm."""
        num_times_node_played = node_played_counter_dict.get(node, 0)
        avg_reward_of_node = avg_reward_dict.get(node, 0)
        num_times_parent_node_played = node_played_counter_dict.get(node.parent_node, 0)
        avg_reward_of_parent_node = avg_reward_dict.get(node.parent_node, 0)

        node_index = min(avg_reward_of_node + self.calc_confidence(num_times_node_played, t),
                         avg_reward_of_parent_node + self.calc_confidence(num_times_parent_node_played, t) +
                         self.v1 * self.rho ** (node.h - 1)) + self.v1 * self.rho ** node.h

        return node_index + self.N * self.v1 / self.v2 * self.v1 * self.rho ** node.h

    def calc_confidence(self, num_times_node_played, t):
        """Calculate the confidence bound."""
        if num_times_node_played == 0:
            return float('inf')
        return sqrt(2 * math.log(t) / num_times_node_played)
    
    def get_leaves_count_arr(self):
        """Get the array of leaves count over time."""
        return self.leaves_count_arr
    
    def run_algorithm(self):
        """Run the HD-ACC-UCB algorithm and return results including leaves count."""
        total_reward_arr, regret_arr, played_arms_arr = super().run_algorithm()
        return total_reward_arr, regret_arr, played_arms_arr, self.leaves_count_arr
