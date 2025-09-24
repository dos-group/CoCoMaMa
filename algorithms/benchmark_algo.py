import numpy as np
import random

import ProblemModel
from collections import Counter



def find_node_containing_context(context, leaves):
    for leaf in leaves:
        if leaf.contains_context(context):
            return leaf


"""
This class represents a greedy benchmark that picks the K arms with highest means.
"""


class Benchmark:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, budget):
        self.num_rounds = problem_model.num_rounds
        self.budget = budget  # Keep budget for regret calculation, but don't use it for selection
        self.problem_model = problem_model

    def run_algorithm(self):
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        played_arms_arr = []
        uniquely_best_arms_arr = []  # Keep track of uniquely best arms

        for t in range(1, self.num_rounds + 1):
            available_arms = self.problem_model.get_available_arms(t)
            
            # Sort arms by their true means
            available_arms_with_means = [(arm, arm.true_mean) for arm in available_arms]
            available_arms_with_means.sort(key=lambda x: x[1], reverse=True)
            
            # Find the highest mean
            highest_mean = available_arms_with_means[0][1]
            
            # Count how many arms achieve the highest mean
            num_best_arms = sum(1 for _, mean in available_arms_with_means if mean == highest_mean)
            
            # Store uniquely best arm if there is one
            if num_best_arms == 1:
                uniquely_best_arms_arr.append([available_arms_with_means[0][0]])
            else:
                uniquely_best_arms_arr.append([])
            
            # Select all arms that achieve the highest mean
            slate = [arm for arm, mean in available_arms_with_means if mean == highest_mean]
            
            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards)
            regret_arr[t - 1] = self.problem_model.get_regret(t, self.budget, slate)
            played_arms_arr.append(slate)

        return total_reward_arr, regret_arr, played_arms_arr, uniquely_best_arms_arr


