import math
from math import sqrt

import numpy as np

import ProblemModel
from algorithms import random_algo
from UcbNode import UcbNode


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


"""
Adaptive Contextual Combinatorial Upper Confidence Bound (ACC-UCB) Algorithm

This implementation is based on the work by A. Nika, S. Elahi and C. Tekin:
"Contextual combinatorial volatile multi-armed bandit with adaptive discretization"
23rd International Conference on Artificial Intelligence and Statistics (AISTATS), 2024.

Original implementation available at: https://github.com/Bilkent-CYBORG/ACC-UCB

The algorithm handles contextual combinatorial volatile multi-armed bandit problems
with adaptive discretization of the context space using a binary tree structure.
"""


class ACCUCB:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, v1, v2, N, rho, budget, initial_hypercube):
        """
        Initialize the ACC-UCB algorithm.
        
        Args:
            problem_model: The problem model containing arms and reward information
            v1: First parameter for the confidence bound calculation
            v2: Second parameter for the confidence bound calculation  
            N: Number of dimensions for the binary tree (must be 2)
            rho: Parameter controlling the splitting threshold
            budget: Number of arms to select in each round
            initial_hypercube: Initial hypercube defining the context space
        """
        self.initial_hypercube = initial_hypercube
        if N != 2:
            print('ACC-UCB ONLY works when N = 2')
            exit(1)
        self.N = N
        self.num_rounds = problem_model.num_rounds
        self.budget = budget
        self.rho = rho
        self.v2 = v2
        self.v1 = v1
        self.problem_model = problem_model

    def run_algorithm(self):
        """
        Run the ACC-UCB algorithm for the specified number of rounds.
        
        Returns:
            tuple: (total_reward_arr, regret_arr, played_agents_arr, leaves_count_arr)
                - total_reward_arr: Array of total rewards obtained in each round
                - regret_arr: Array of regret values for each round
                - played_agents_arr: List of selected arms for each round
                - leaves_count_arr: Array of number of leaves in the tree for each round
        """
        self.num_rounds = self.problem_model.num_rounds
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        played_agents_arr = []
        leaves_count_arr = []
        leaves = [UcbNode(None, 0, self.initial_hypercube)]
        node_played_counter_dict = {}
        avg_reward_dict = {}

        for t in range(1, self.num_rounds + 1):
            available_arms = self.problem_model.get_available_arms(t)
            index_list = np.zeros(len(available_arms))
            i = 0
            # Only log every 5% of progress
            if t % max(1, self.num_rounds // 20) == 0:
                progress = (t / self.num_rounds) * 100
                avg_reward = np.mean(total_reward_arr[:t])
                avg_regret = np.mean(regret_arr[:t])
                print(f'[ACCUCB] Progress: {progress:.1f}% | Time t = {t} | Available arms = {len(available_arms)} | Leaves = {len(leaves)} | Avg Reward = {avg_reward:.3f} | Avg Regret = {avg_regret:.3f}')
            leaves_count_arr.append(len(leaves))


            # Check if only root node is available
            if len(leaves) == 1:
                arm_indices_to_play = random_algo.sample(range(len(available_arms)), self.budget)
            else:
                for available_arm in available_arms:
                    node = find_node_containing_context(available_arm.context, leaves)
                    index_list[i] = self.get_arm_index(node, node_played_counter_dict, avg_reward_dict, t)
                    i += 1

                arm_indices_to_play = self.problem_model.oracle(self.budget, index_list)

            selected_nodes = set()
            slate = []
            for index in arm_indices_to_play:
                selected_arm = available_arms[index]
                slate.append(selected_arm)
                node = find_node_containing_context(selected_arm.context, leaves)
                selected_nodes.add(node)
                # Initialize the node in the counter dictionary if it doesn't exist
                if node not in node_played_counter_dict:
                    node_played_counter_dict[node] = 0
                if node not in avg_reward_dict:
                    avg_reward_dict[node] = 0
            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects
            played_agents_arr.append(slate)

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards)
            regret_arr[t - 1] = self.problem_model.get_regret(t, self.budget, slate)

            # Update the counters
            for reward in rewards:
                node_with_context = find_node_containing_context(reward.context, selected_nodes)
                new_counter = node_played_counter_dict[node_with_context] = node_played_counter_dict.get(
                    node_with_context, 0) + 1
                avg_reward_dict[node_with_context] = (avg_reward_dict.get(node_with_context, 0) * (
                            new_counter - 1) + reward.quality) / new_counter

            for selected_node in selected_nodes:
                # Ensure the node exists in the counter dictionary before accessing it
                if selected_node not in node_played_counter_dict:
                    node_played_counter_dict[selected_node] = 0
                if selected_node not in avg_reward_dict:
                    avg_reward_dict[selected_node] = 0
                    
                # Split the node if needed
                if self.calc_confidence(
                        node_played_counter_dict[selected_node], t) <= self.v1 * self.rho ** selected_node.h:
                    produced_nodes = selected_node.reproduce()
                    leaves.remove(selected_node)
                    leaves.extend(produced_nodes)
        return total_reward_arr, regret_arr, played_agents_arr, leaves_count_arr

    def get_arm_index(self, node, node_played_counter_dict, avg_reward_dict, t):
        """
        Calculate the UCB index for a given node.
        
        Args:
            node: The UCB node to calculate the index for
            node_played_counter_dict: Dictionary mapping nodes to play counts
            avg_reward_dict: Dictionary mapping nodes to average rewards
            t: Current time step
            
        Returns:
            float: The UCB index value for the node
        """
        num_times_node_played = node_played_counter_dict.get(node, 0)
        avg_reward_of_node = avg_reward_dict.get(node, 0)
        num_times_parent_node_played = node_played_counter_dict.get(node.parent_node, 0)
        avg_reward_of_parent_node = avg_reward_dict.get(node.parent_node, 0)

        node_index = min(avg_reward_of_node + self.calc_confidence(num_times_node_played, t),
                         avg_reward_of_parent_node + self.calc_confidence(num_times_parent_node_played, t) +
                         self.v1 * self.rho ** (node.h - 1)) + self.v1 * self.rho ** node.h

        return node_index + self.N * self.v1 / self.v2 * self.v1 * self.rho ** node.h

    def calc_confidence(self, num_times_node_played, t):
        """
        Calculate the confidence bound for a node based on the number of times it was played.
        
        Args:
            num_times_node_played: Number of times the node has been played
            t: Current time step
            
        Returns:
            float: The confidence bound value
        """
        if num_times_node_played == 0:
            return float('inf')
        return sqrt(2 * math.log(t) / num_times_node_played)


