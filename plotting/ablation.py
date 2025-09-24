"""
Ablation study plotting utilities with improved type safety and configurability.

This module provides functions for creating plots that compare different parameter
values in ablation studies, with support for error bars and configurable styling.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class AblationPlotConfig:
    """Configuration for ablation study plots."""
    # Algorithm and parameter information
    algorithm_name: str
    parameter_name: str
    parameter_values: List[Union[str, float, int]]
    
    # Plot styling
    colors: Optional[List[str]] = None
    line_width: float = 2.0
    error_bar_alpha: float = 0.3
    grid_alpha: float = 0.3
    
    # Figure settings
    regret_figsize: Tuple[int, int] = (15, 12)
    reward_figsize: Tuple[int, int] = (15, 12)
    leaves_figsize: Tuple[int, int] = (15, 12)
    
    # Labels and titles
    xlabel: str = "Arriving task $(t)$"
    regret_ylabel: str = "Cumulative regret up to $t$"
    reward_ylabel: str = "Average task reward up to $t$"
    leaves_ylabel: str = "Average leaves up to $t$"
    
    # Legend formatting
    legend_template: str = "{algorithm_name} ({param_name}={param_value})"
    
    def __post_init__(self):
        """Set default colors if not provided."""
        if self.colors is None:
            self.colors = ['red', 'green', 'blue', 'cyan', 'orange', 'purple', 'brown', 'pink', 'gray']


def extract_parameter_value(key: str, parameter_name: str) -> Optional[str]:
    """
    Extract parameter value from a results key.
    
    Args:
        key: The results key (e.g., 'streaming_neural_cocoma_theta_2.0_avg_regret')
        parameter_name: The parameter name (e.g., 'theta')
    
    Returns:
        The parameter value as a string, or None if not found
    """
    try:
        # Look for pattern: algorithm_param_name_value_metric
        parts = key.split('_')
        param_idx = None
        for i, part in enumerate(parts):
            if part == parameter_name and i + 1 < len(parts):
                param_idx = i + 1
                break
        
        if param_idx is not None:
            return parts[param_idx]
        
        # Fallback: look for pattern: param_name_value
        if f'{parameter_name}_' in key:
            return key.split(f'{parameter_name}_')[-1].split('_')[0]
            
    except (IndexError, ValueError):
        pass
    
    return None


def plot_with_error_bars(ax: plt.Axes, x_data: List[int], y_data: List[float], 
                        std_data: Optional[List[float]], color: str, label: str, 
                        line_width: float = 2.0, error_alpha: float = 0.3, 
                        num_std_to_show: int = 5) -> None:
    """
    Plot a line with discrete error bars.
    
    Args:
        ax: Matplotlib axes object
        x_data: X-axis data points
        y_data: Y-axis data points (mean values)
        std_data: Standard deviation data for error bars (optional)
        color: Line color
        label: Line label for legend
        line_width: Width of the main line
        error_alpha: Transparency of error bars
        num_std_to_show: Number of error bars to show (controls downsampling)
    """
    ax.plot(x_data, y_data, color=color, label=label, linewidth=line_width)
    
    if std_data is not None and len(std_data) == len(y_data):
        # Create downsampled data for error bars (same logic as existing plotting functions)
        x_err = []
        y_err = []
        yerr = []
        
        for i in range(len(std_data)):
            if i == 0 or i % int(len(std_data) / num_std_to_show) != 0 and i != len(std_data) - 1:
                continue
            x_err.append(x_data[i])
            y_err.append(y_data[i])
            yerr.append(std_data[i])
        
        # Plot discrete error bars
        if len(x_err) > 0:
            ax.errorbar(x_err, y_err, yerr=yerr, color=color, alpha=error_alpha, 
                       fmt='none', capsize=3, capthick=1)


def plot_ablation_regret(aggregated_results: Dict[int, Dict[str, List[float]]], 
                        budgets: List[int], 
                        num_rounds: int, 
                        results_dir: str,
                        config: AblationPlotConfig,
                        num_std_to_show: int = 5) -> None:
    """
    Create cumulative regret plots for ablation study.
    
    Args:
        aggregated_results: Dictionary mapping budget -> metric_name -> values
        budgets: List of budget values to plot
        num_rounds: Number of rounds (x-axis range)
        results_dir: Directory to save the plot
        config: Configuration for the plot styling and content
    """
    plt.figure(figsize=config.regret_figsize)
    num_plots = len(budgets)
    rows = int(num_plots / 2 + (num_plots % 2))
    cols = 2

    for i, budget in enumerate(budgets):
        ax = plt.subplot(rows, cols, i + 1)
        budget_data = aggregated_results.get(budget, {})
        
        color_idx = 0
        
        for key, value in budget_data.items():
            # Check if this is a regret metric for our parameter
            if (config.parameter_name in key and 
                'avg_regret' in key and 
                config.algorithm_name in key):
                
                # Extract parameter value
                param_value = extract_parameter_value(key, config.parameter_name)
                if param_value is None:
                    continue
                
                # Get standard deviation data if available
                std_key = key.replace('avg_regret', 'std_regret')
                std_data = budget_data.get(std_key)
                
                # Create label
                label = config.legend_template.format(
                    algorithm_name=config.algorithm_name.replace('_', '-').title(),
                    param_name=config.parameter_name,
                    param_value=param_value
                )
                
                # Plot with error bars
                plot_with_error_bars(
                    ax, range(1, num_rounds + 1), value, std_data,
                    config.colors[color_idx % len(config.colors)], label,
                    config.line_width, config.error_bar_alpha, num_std_to_show
                )
                color_idx += 1
        
        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.regret_ylabel)
        ax.set_title(f'Cumulative Regret (b = {budget})')
        if color_idx > 0:  # Only show legend if we have data
            ax.legend()
        ax.grid(True, alpha=config.grid_alpha)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{results_dir}/ablation_regret.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.close()


def plot_ablation_reward(aggregated_results: Dict[int, Dict[str, List[float]]], 
                        budgets: List[int], 
                        num_rounds: int, 
                        results_dir: str,
                        config: AblationPlotConfig,
                        num_std_to_show: int = 5) -> None:
    """
    Create average reward plots for ablation study.
    
    Args:
        aggregated_results: Dictionary mapping budget -> metric_name -> values
        budgets: List of budget values to plot
        num_rounds: Number of rounds (x-axis range)
        results_dir: Directory to save the plot
        config: Configuration for the plot styling and content
    """
    plt.figure(figsize=config.reward_figsize)

    num_plots = len(budgets)
    rows = int(num_plots / 2 + (num_plots % 2))
    cols = 2

    for i, budget in enumerate(budgets):
        ax = plt.subplot(rows, cols, i + 1)
        budget_data = aggregated_results.get(budget, {})
        
        color_idx = 0
        
        for key, value in budget_data.items():
            # Check if this is a regret metric for our parameter
            if (config.parameter_name in key and 
                'avg_reward' in key and 
                config.algorithm_name in key):
                
                # Extract parameter value
                param_value = extract_parameter_value(key, config.parameter_name)
                if param_value is None:
                    continue
                
                # Get standard deviation data if available
                std_key = key.replace('avg_reward', 'std_reward')
                std_data = budget_data.get(std_key)
                
                # Create label
                label = config.legend_template.format(
                    algorithm_name=config.algorithm_name.replace('_', '-').title(),
                    param_name=config.parameter_name,
                    param_value=param_value
                )
                
                # Plot with error bars
                plot_with_error_bars(
                    ax, range(1, num_rounds + 1), value, std_data,
                    config.colors[color_idx % len(config.colors)], label,
                    config.line_width, config.error_bar_alpha, num_std_to_show
                )
                color_idx += 1
        
        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.reward_ylabel)
        ax.set_title(f'Average Reward (b = {budget})')
        if color_idx > 0:  # Only show legend if we have data
            ax.legend()
        ax.grid(True, alpha=config.grid_alpha)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{results_dir}/ablation_reward.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.close()


def plot_ablation_leaves(aggregated_results: Dict[int, Dict[str, List[float]]], 
                        budgets: List[int], 
                        num_rounds: int, 
                        results_dir: str,
                        config: AblationPlotConfig,
                        num_std_to_show: int = 5) -> None:
    """
    Create average leaves plots for ablation study.
    
    Args:
        aggregated_results: Dictionary mapping budget -> metric_name -> values
        budgets: List of budget values to plot
        num_rounds: Number of rounds (x-axis range)
        results_dir: Directory to save the plot
        config: Configuration for the plot styling and content
    """
    
    plt.figure(figsize=config.leaves_figsize)

    num_plots = len(budgets)
    rows = int(num_plots / 2 + (num_plots % 2))
    cols = 2

    for i, budget in enumerate(budgets):
        ax = plt.subplot(rows, cols, i + 1)
        budget_data = aggregated_results.get(budget, {})
        
        color_idx = 0
        
        for key, value in budget_data.items():
            # Check if this is a regret metric for our parameter
            if (config.parameter_name in key and 
                'avg_leaves' in key and 
                config.algorithm_name in key):
                
                # Extract parameter value
                param_value = extract_parameter_value(key, config.parameter_name)
                if param_value is None:
                    continue
                
                # Get standard deviation data if available
                std_key = key.replace('avg_leaves', 'std_leaves')
                std_data = budget_data.get(std_key)
                
                # Create label
                label = config.legend_template.format(
                    algorithm_name=config.algorithm_name.replace('_', '-').title(),
                    param_name=config.parameter_name,
                    param_value=param_value
                )
                
                # Plot with error bars
                plot_with_error_bars(
                    ax, range(1, num_rounds + 1), value, std_data,
                    config.colors[color_idx % len(config.colors)], label,
                    config.line_width, config.error_bar_alpha, num_std_to_show
                )
                color_idx += 1
        
        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.leaves_ylabel)
        ax.set_title(f'Average Leaves (b = {budget})')
        if color_idx > 0:  # Only show legend if we have data
            ax.legend()
        ax.grid(True, alpha=config.grid_alpha)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{results_dir}/ablation_leaves.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.close()

def create_ablation_plots(aggregated_results: Dict[int, Dict[str, List[float]]], 
                         budgets: List[int], 
                         num_rounds: int, 
                         results_dir: str,
                         algorithm_name: str,
                         parameter_name: str,
                         parameter_values: List[Union[str, float, int]],
                         num_std_to_show: int = 5,
                         **kwargs) -> None:
    """
    Create all ablation study plots with a single function call.
    
    Args:
        aggregated_results: Dictionary mapping budget -> metric_name -> values
        budgets: List of budget values to plot
        num_rounds: Number of rounds (x-axis range)
        results_dir: Directory to save the plots
        algorithm_name: Name of the algorithm being tested
        parameter_name: Name of the parameter being varied
        parameter_values: List of parameter values being tested
        num_std_to_show: Number of error bars to show (controls downsampling)
        **kwargs: Additional configuration options for AblationPlotConfig
    """
    config = AblationPlotConfig(
        algorithm_name=algorithm_name,
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        **kwargs
    )
    
    plot_ablation_regret(aggregated_results, budgets, num_rounds, results_dir, config, num_std_to_show)
    plot_ablation_reward(aggregated_results, budgets, num_rounds, results_dir, config, num_std_to_show)
    plot_ablation_leaves(aggregated_results, budgets, num_rounds, results_dir, config, num_std_to_show)