import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import ProblemModel
from algorithms import random_algo
from UcbNode import UcbNode


def find_node_containing_context(context, leaves):
    for leaf in leaves:
        if leaf.contains_context(context):
            return leaf
        
def update(x, n, mean, M2, M3):
        delta = x - mean
        delta_n = delta / n
        term1 = delta * delta_n * (n - 1)

        mean += delta_n
        M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2
        M2 += term1

        return mean, M2, M3

def update_covariance(n, mean_x, mean_y, C_xy, x, y):
    delta_x = x - mean_x
    delta_y = y - mean_y
    m2_y = delta_y * delta_y / n * (n - 1)


    mean_x += delta_x / n
    mean_y += delta_y / n

    C_xy += delta_x * (y - mean_y)

    return mean_x, mean_y, C_xy, m2_y

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


class Neural_CoCoMaMa:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, v1, v2, N, rho, budget, initial_hypercube):
        self.initial_hypercube = initial_hypercube
        if N != 2:
            print('Neural_CoCoMaMa ONLY works when N = 2')
            exit(1)
        
        self.problem_model = problem_model
        
        # Get available arms to determine input dimension
        available_arms = self.problem_model.get_available_arms(1)
        if not available_arms:
            raise ValueError("No available arms found to determine input dimension")
        self.N = N
        self.num_rounds = problem_model.num_rounds
        self.budget = budget
        self.rho = rho
        self.v2 = v2
        self.v1 = v1
        # multiplier added to average reward variance to decide when a node should be split
        self.theta = 4
        # Initialize neural networks
        self.input_dim = len(available_arms[0].context)
        self.single_reward_net = SingleRewardNet(self.input_dim)
        # Initialize optimizers
        self.single_optimizer = optim.Adam(self.single_reward_net.parameters())
        # Initialize loss functions
        self.single_criterion = nn.MSELoss()



    def train_single_reward_net(self, contexts, rewards):
        """Train the single reward neural network."""
        contexts = torch.FloatTensor(contexts)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        
        self.single_optimizer.zero_grad()
        predictions = self.single_reward_net(contexts)
        loss = self.single_criterion(predictions, rewards)
        loss.backward()
        self.single_optimizer.step()


    def run_algorithm(self):
        self.num_rounds = self.problem_model.num_rounds
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        played_agents_arr = []
        leaves_count_arr = []
        leaves = [UcbNode(None, 0, self.initial_hypercube)]
        node_played_counter_dict = {}
        avg_reward_dict = {}
        m2_reward_dict = {}
        cov_context_reward_dict = {}
        avg_context_dict = {}
        
        # New dictionaries to store metrics for last 100 rounds
        last_100_metrics = {
            'task_agent_rewards': [{} for _ in range(100)],  # Array of agent->reward mappings for last 100 rounds
            'task_agent_counts': [{} for _ in range(100)],   # Array of agent->count mappings for last 100 rounds
        }

        # Store metrics at 10% and 50% completion
        leaves_10 = None
        leaves_50 = None
        node_played_counter_dict_10 = None
        node_played_counter_dict_50 = None
        m2_reward_dict_10 = None
        m2_reward_dict_50 = None
        avg_reward_dict_10 = None
        avg_reward_dict_50 = None

        for t in range(1, self.num_rounds + 1):
            # Store metrics at 10% and 50% completion
            if t == int(self.num_rounds * 0.1):
                leaves_10 = leaves.copy()
                node_played_counter_dict_10 = node_played_counter_dict.copy()
                m2_reward_dict_10 = m2_reward_dict.copy()
                avg_reward_dict_10 = avg_reward_dict.copy()
            elif t == int(self.num_rounds * 0.5):
                leaves_50 = leaves.copy()
                node_played_counter_dict_50 = node_played_counter_dict.copy()
                m2_reward_dict_50 = m2_reward_dict.copy()
                avg_reward_dict_50 = avg_reward_dict.copy()

            available_arms = self.problem_model.get_available_arms(t)
            index_list = np.zeros(len(available_arms))
            i = 0
            # Only log every 5% of progress
            if t % max(1, self.num_rounds // 20) == 0:
                progress = (t / self.num_rounds) * 100
                avg_reward = np.mean(total_reward_arr[:t])
                avg_regret = np.mean(regret_arr[:t])
                print(f'[Neural_CoCoMaMa] Progress: {progress:.1f}% | Time t = {t} | Available arms = {len(available_arms)} | Leaves = {len(leaves)} | Avg Reward = {avg_reward:.3f} | Avg Regret = {avg_regret:.3f}')
            leaves_count_arr.append(len(leaves))

            # Check if only root node is available
            if len(leaves) == 1:
                arm_indices_to_play = random_algo.sample(range(len(available_arms)), self.budget)
            else:
                for available_arm in available_arms:
                    node = find_node_containing_context(available_arm.context, leaves)
                    index_list[i] = self.get_arm_index(node, node_played_counter_dict, t, available_arm.context)
                    i += 1
                    if t > self.num_rounds - 100:
                        last_100_metrics["task_agent_rewards"][self.num_rounds-t][available_arm]=avg_reward_dict.get(node, 0)
                        last_100_metrics['task_agent_counts'][self.num_rounds-t][available_arm]=node_played_counter_dict.get(node,0)

                arm_indices_to_play = self.problem_model.oracle(self.budget, index_list)

            selected_nodes = set()
            slate = []
            for index in arm_indices_to_play:
                selected_arm = available_arms[index]
                slate.append(selected_arm)
                selected_nodes.add(find_node_containing_context(selected_arm.context, leaves))
            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects
            played_agents_arr.append(slate)

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards)
            regret_arr[t - 1] = self.problem_model.get_regret(t, self.budget, slate)

            # Train neural networks
            contexts = [reward.context for reward in rewards]
            single_rewards = [reward.quality for reward in rewards]
            self.train_single_reward_net(contexts, single_rewards)
            # Update the counters
            for reward in rewards:
                node_with_context = find_node_containing_context(reward.context, selected_nodes)
                new_counter = node_played_counter_dict[node_with_context] = node_played_counter_dict.get(
                    node_with_context, 0) + 1
                # Update the reward statistics
                old_mean = avg_reward_dict.get(node_with_context, 0)
                old_cov = cov_context_reward_dict.get(node_with_context, np.zeros(len(reward.context)))
                old_context_mean = avg_context_dict.get(node_with_context, np.zeros(len(reward.context)))
                avg_reward_dict[node_with_context] = (avg_reward_dict.get(node_with_context, 0) * (
                            new_counter - 1) + reward.quality) / new_counter
                mean_x, mean_y, C_xy, m2_y = update_covariance(new_counter, old_context_mean, old_mean, old_cov, reward.context, reward.quality)
                cov_context_reward_dict[node_with_context] = C_xy
                avg_reward_dict[node_with_context] = mean_y
                m2_reward_dict[node_with_context]= m2_y
                avg_context_dict[node_with_context] = mean_x



            for selected_node in selected_nodes:
                selected_node_reward_variance = m2_reward_dict.get(selected_node, 0) / node_played_counter_dict.get(selected_node, 1)
                # selected_node_reward_variance = (m2_reward_dict.get(selected_node, 0) / node_played_counter_dict.get(selected_node, 1)) * self.rho ** selected_node.h
                split_threshold = self.calc_reward_variance_split_threshold(leaves, m2_reward_dict, node_played_counter_dict)

                selected_node_played_count = node_played_counter_dict.get(selected_node,0)
                parent_node_played_count = node_played_counter_dict.get(selected_node.parent_node,0)
                # Split the node if needed
                if (split_threshold < selected_node_reward_variance and parent_node_played_count < selected_node_played_count) or self.calc_confidence(selected_node_played_count, t)<= self.v1 * self.rho ** selected_node.h:
                    produced_nodes = selected_node.reproduce_informed(cov_context_reward_dict[selected_node], avg_context_dict[selected_node])
                    leaves.remove(selected_node)
                    leaves.extend(produced_nodes)

        # Calculate final metrics for last 100 rounds
        final_metrics = {
            'task_agent_avg_rewards': last_100_metrics['task_agent_rewards'],
            'task_agent_counts': last_100_metrics['task_agent_counts'],
            'leaves': leaves,
            'leaf_counts': node_played_counter_dict,
            'leaf_variances': m2_reward_dict,
            'leaf_rewards': avg_reward_dict,
            'leaves_10': leaves_10 if leaves_10 is not None else leaves,
            'leaf_counts_10': node_played_counter_dict_10 if node_played_counter_dict_10 is not None else node_played_counter_dict,
            'leaf_variances_10': m2_reward_dict_10 if m2_reward_dict_10 is not None else m2_reward_dict,
            'leaf_rewards_10': avg_reward_dict_10 if avg_reward_dict_10 is not None else avg_reward_dict,
            'leaves_50': leaves_50 if leaves_50 is not None else leaves,
            'leaf_counts_50': node_played_counter_dict_50 if node_played_counter_dict_50 is not None else node_played_counter_dict,
            'leaf_variances_50': m2_reward_dict_50 if m2_reward_dict_50 is not None else m2_reward_dict,
            'leaf_rewards_50': avg_reward_dict_50 if avg_reward_dict_50 is not None else avg_reward_dict,
        }

        return total_reward_arr, regret_arr, played_agents_arr, leaves_count_arr, final_metrics

    def get_arm_index(self, node, node_played_counter_dict, t, context):
        num_times_node_played = node_played_counter_dict.get(node, 0)
        num_times_parent_node_played = node_played_counter_dict.get(node.parent_node, 0)
        combined_confidence = self.calc_confidence(num_times_node_played + num_times_parent_node_played, t)
        # Estimate overall reward
        with torch.no_grad():
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
            estimated_reward = self.single_reward_net(context_tensor).item()
            return (estimated_reward+combined_confidence)


    def calc_confidence(self, num_times_node_played, t):
        if num_times_node_played == 0:
            return float('inf')
        return sqrt(2 * math.log(t) / num_times_node_played)

    def calc_reward_variance_split_threshold(self, leaves, m2_reward_dict, node_played_counter_dict):
        if len(leaves) == 1:
            return 0
        total_variance = 0
        played_leaves_count = 0
        for leaf in leaves:
            node_played_counter = node_played_counter_dict.get(leaf,0)
            if(node_played_counter > 1):    
                # variance = m2_reward_dict[leaf] / node_played_counter
                total_variance += m2_reward_dict[leaf]
                played_leaves_count += node_played_counter
        if(played_leaves_count <= 1):
            return 0
        weighted_average_variance = total_variance / played_leaves_count
        return weighted_average_variance * self.theta


