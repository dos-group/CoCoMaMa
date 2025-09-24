"""
Streaming version of CoCoMaMa algorithm.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from algorithms.streaming_base import StreamingAlgorithm
from UcbNode import UcbNode
from Hyperrectangle import Hyperrectangle
from Reward import Reward
from Arm import Arm

try:
    from numba import njit  # type: ignore
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

if _NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _find_leaf_index_for_point(point, centers, half_lengths, eps):
        L, D = centers.shape
        for i in range(L):
            outside = False
            for j in range(D):
                if abs(point[j] - centers[i, j]) > half_lengths[i, j] + eps:
                    outside = True
                    break
            if not outside:
                return i
        return -1
else:
    def _find_leaf_index_for_point(point, centers, half_lengths, eps):
        L, D = centers.shape
        for i in range(L):
            outside = False
            for j in range(D):
                if abs(point[j] - centers[i, j]) > half_lengths[i, j] + eps:
                    outside = True
                    break
            if not outside:
                return i
        return -1


class StreamingCoCoMaMa(StreamingAlgorithm):
    """
    Streaming version of CoCoMaMa algorithm.
    """
    
    def __init__(self, problem_model, v1: float, v2: float, N: int, rho: float, 
                 budget: int, initial_hypercube: Hyperrectangle, theta: float = 4.0):
        """
        Initialize streaming CoCoMaMa.
        
        Args:
            problem_model: Streaming problem model
            v1: Algorithm parameter
            v2: Algorithm parameter
            N: Algorithm parameter (must be 2)
            rho: Algorithm parameter
            budget: Budget
            initial_hypercube: Initial hypercube for the algorithm
            theta: Multiplier for variance threshold (default: 4.0)
        """
        super().__init__(problem_model, budget)
        
        if N != 2:
            raise ValueError('CoCoMaMa ONLY works when N = 2')
        
        self.v1 = v1
        self.v2 = v2
        self.N = N
        self.rho = rho
        self.initial_hypercube = initial_hypercube
        self.theta = theta  # multiplier for variance threshold
        
        # Initialize algorithm state
        self.leaves = [UcbNode(None, 0, self.initial_hypercube)]
        self.node_played_counter_dict = {}
        self.avg_reward_dict = {}
        self.m2_reward_dict = {}
        self.cov_context_reward_dict = {}
        self.avg_context_dict = {}
        
        # Metrics tracking
        self.leaves_count_arr = []
        self.last_100_metrics = {
            'task_agent_rewards': [{} for _ in range(100)],
            'task_agent_counts': [{} for _ in range(100)],
        }
        
        # Milestone tracking
        self.leaves_10 = None
        self.leaves_50 = None
        self.node_played_counter_dict_10 = None
        self.node_played_counter_dict_50 = None
        self.m2_reward_dict_10 = None
        self.m2_reward_dict_50 = None
        self.avg_reward_dict_10 = None
        self.avg_reward_dict_50 = None

        # Batch containment acceleration structures
        self._leaf_centers = None  # type: np.ndarray | None
        self._leaf_half_lengths = None  # type: np.ndarray | None
        self._eps = 1e-10
        self._rebuild_leaf_arrays()
    
    def select_arms(self, available_arms: List[Arm]) -> List[Arm]:
        """Select arms using CoCoMaMa algorithm."""
        # Select top K arms
        if len(available_arms) <= self.budget:
            return available_arms
        
        # Calculate index values for each arm
        arm_indices = []
        for arm in available_arms:
            context = arm.context
            node = self._find_node_containing_context(context, self.leaves)
            
            arm_index = self._get_arm_index(node, len(self.leaves_count_arr) + 1)
            arm_indices.append(arm_index)
        
        # Get indices of top K arms
        top_indices = np.argsort(arm_indices)[-self.budget:]
        return [available_arms[i] for i in top_indices]
    
    def update(self, selected_arms: List[Arm], 
               rewards: List[Reward], round_num: int):
        """Update the algorithm with observed rewards."""
        # Update statistics for each selected arm and collect unique nodes
        selected_nodes = set()
        for arm, reward in zip(selected_arms, rewards):
            unique_id = arm.unique_id
            reward_value = reward.quality
            context = arm.context

            node = self._find_node_containing_context(context, self.leaves)
            selected_nodes.add(node)

            # Update node statistics
            self._update_node_statistics(node, context, reward_value, unique_id)

        # Iterate over unique selected nodes and split if necessary
        for node in selected_nodes:
            if self._should_split_node(node, round_num):
                self._split_node(node)
        
        # Update metrics
        self._update_metrics()
    
    def _find_node_containing_context(self, context: np.ndarray, leaves: List[UcbNode]) -> UcbNode:
        """Find the leaf node containing the given context."""
        # Fast batch path using stacked arrays and optional Numba
        if self._leaf_centers is None or self._leaf_half_lengths is None or len(leaves) == 0:
            return leaves[0] if leaves else None
        point = np.asarray(context, dtype=self._leaf_centers.dtype)
        idx = _find_leaf_index_for_point(point, self._leaf_centers, self._leaf_half_lengths, self._eps)
        if idx >= 0:
            return leaves[int(idx)]
        # Fallback scan (shouldn't happen)
        for leaf in leaves:
            if leaf.contains_context(point):
                return leaf
        return leaves[0] if leaves else None
    


    def _get_arm_index(self, node, t):
        node_key = id(node)

        num_times_node_played = self.node_played_counter_dict.get(node_key, 0)
        avg_reward_of_node = self.avg_reward_dict.get(node_key, 0)
        num_times_parent_node_played = self.node_played_counter_dict.get(id(node.parent_node), 0)
        avg_reward_of_parent_node = self.avg_reward_dict.get(id(node.parent_node), 0)

        combined_confidence = self._calc_confidence(num_times_node_played + num_times_parent_node_played, t)

        return max(avg_reward_of_node + combined_confidence, avg_reward_of_parent_node+combined_confidence)


    def _calc_confidence(self, num_times_node_played, t):
        if num_times_node_played == 0:
            return float('inf')
        return np.sqrt(2 * np.log(t) / num_times_node_played)
    
    def _update_node_statistics(self, node: UcbNode, context: np.ndarray, reward: float, unique_id: int):
        """Update statistics for a node."""
        node_key = id(node)
        
        # Initialize if needed
        if node_key not in self.node_played_counter_dict:
            self.node_played_counter_dict[node_key] = 0
            self.avg_reward_dict[node_key] = 0.0
            self.m2_reward_dict[node_key] = 0.0
            self.cov_context_reward_dict[node_key] = np.zeros(len(context))
            self.avg_context_dict[node_key] = np.zeros(len(context))
        
        # Update counters
        old_n = self.node_played_counter_dict[node_key]
        self.node_played_counter_dict[node_key] += 1
        
        # Update running statistics using Welford's algorithm
        old_avg_reward = self.avg_reward_dict[node_key]
        old_avg_context = self.avg_context_dict[node_key].copy()
        
        # Update reward statistics
        delta_reward = reward - old_avg_reward
        self.avg_reward_dict[node_key] += delta_reward / (old_n + 1)
        self.m2_reward_dict[node_key] += delta_reward * delta_reward / (old_n + 1) * old_n
        
        # Update context statistics
        delta_context = context - old_avg_context
        self.avg_context_dict[node_key] += delta_context / (old_n + 1)
        self.cov_context_reward_dict[node_key] += delta_context * (reward - old_avg_reward)
    
    def _should_split_node(self, node: UcbNode, round_num: int) -> bool:
        """Check if a node should be split."""
        node_key = id(node)
        n = self.node_played_counter_dict.get(node_key, 1)
        
        if n < 2:
            return False
        
        # Calculate variance
        variance = self.m2_reward_dict[node_key] / n
        variance_split_threshold = self._calc_reward_variance_split_threshold()
        parent_node_played_count = self.node_played_counter_dict.get(id(node.parent_node),0)
        confidence = self._calc_confidence(n, round_num)
        ucb_threshold = self.v1 * self.rho ** node.h
        
        # Check if variance exceeds threshold
        return variance > variance_split_threshold and n > parent_node_played_count or confidence <= ucb_threshold
    
    def _calc_reward_variance_split_threshold(self):
        if len(self.leaves) == 1:
            return 0
        total_m2_reward = 0
        played_leaves_count = 0
        for leaf in self.leaves:
            node_played_counter = self.node_played_counter_dict.get(id(leaf),0)
            if(node_played_counter > 1):    
                # variance = m2_reward_dict[leaf] / node_played_counter
                total_m2_reward += self.m2_reward_dict[id(leaf)]
                played_leaves_count += node_played_counter
        if(played_leaves_count <= 1):
            return 0
        weighted_average_variance = total_m2_reward / played_leaves_count
        return weighted_average_variance * self.theta
    
    def _split_node(self, node: UcbNode):
        """Split a node into two children."""
        produced_nodes = node.reproduce_informed(self.cov_context_reward_dict[id(node)], self.avg_context_dict[id(node)])
        self.leaves.remove(node)
        self.leaves.extend(produced_nodes)
        self._rebuild_leaf_arrays()
    
    def _update_metrics(self):
        """Update algorithm metrics."""
        # Update leaves count
        self.leaves_count_arr.append(len(self.leaves))
        
        # Update milestone tracking
        current_round = len(self.leaves_count_arr)
        if current_round == int(0.1 * self.num_rounds):
            self.leaves_10 = self.leaves.copy()
            self.node_played_counter_dict_10 = self.node_played_counter_dict.copy()
            self.m2_reward_dict_10 = self.m2_reward_dict.copy()
            self.avg_reward_dict_10 = self.avg_reward_dict.copy()
        elif current_round == int(0.5 * self.num_rounds):
            self.leaves_50 = self.leaves.copy()
            self.node_played_counter_dict_50 = self.node_played_counter_dict.copy()
            self.m2_reward_dict_50 = self.m2_reward_dict.copy()
            self.avg_reward_dict_50 = self.avg_reward_dict.copy()
    
    def run_algorithm(self) -> Tuple[np.ndarray, np.ndarray, List[List[Any]], List[int], Dict[str, Any]]:
        """
        Run the streaming CoCoMaMa algorithm.
        
        Returns:
            Tuple of (rewards, regrets, played_arms, leaves_count, metrics)
        """
        rewards, regrets, played_arms = super().run_algorithm()
        
        # Prepare metrics
        metrics = {
            'leaves_10': self.leaves_10,
            'leaves_50': self.leaves_50,
            'node_played_counter_dict_10': self.node_played_counter_dict_10,
            'node_played_counter_dict_50': self.node_played_counter_dict_50,
            'm2_reward_dict_10': self.m2_reward_dict_10,
            'm2_reward_dict_50': self.m2_reward_dict_50,
            'avg_reward_dict_10': self.avg_reward_dict_10,
            'avg_reward_dict_50': self.avg_reward_dict_50,
            'last_100_metrics': self.last_100_metrics
        }
        
        return rewards, regrets, played_arms, self.leaves_count_arr, metrics

    def _rebuild_leaf_arrays(self) -> None:
        """Rebuild stacked centers and half-length arrays for fast batch containment."""
        if not self.leaves:
            self._leaf_centers = None
            self._leaf_half_lengths = None
            return
        dim = self.leaves[0].hyperrectangle.get_dimension()
        centers = np.empty((len(self.leaves), dim), dtype=np.float64)
        half_lengths = np.empty((len(self.leaves), dim), dtype=np.float64)
        for i, leaf in enumerate(self.leaves):
            hr = leaf.hyperrectangle
            centers[i, :] = np.asarray(hr.center, dtype=np.float64)
            # Hyperrectangle maintains half_length; compute if missing
            if hasattr(hr, 'half_length'):
                half_lengths[i, :] = np.asarray(hr.half_length, dtype=np.float64)
            else:
                half_lengths[i, :] = np.asarray(hr.length, dtype=np.float64) * 0.5
        self._leaf_centers = np.ascontiguousarray(centers)
        self._leaf_half_lengths = np.ascontiguousarray(half_lengths)
