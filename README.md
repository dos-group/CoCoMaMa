# CoCoMaMa: Combinatorial Contextual Volatile Multi-Armed Bandit for Agent Routing

This repository contains the experiments for the CoCoMaMa paper: "CoCoMaMa: Contextual Combinatorial Multi-Armed Bandit Router for Multi-Agent Systems with Volatile Arms".

## Abstract

Agentic Large Language Models (LLMs) are designed for specialized objectives using fine-tuning, prompting techniques, and tool calling to outperform general-purpose models in their expert domains. Standardization efforts like the Agent2Agent Protocol could drastically increase the number and heterogeneity of experts available via the Web. A router is required to find the best agent for any given task. However, existing LLM routing methods use a fixed-sized pool of models and often rely on offline training data such as benchmarks. We propose CoCoMaMa and Neural-CoCoMaMa, a combinatorial contextual volatile multi-armed bandit approach that leverages similarities between tasks and agents by learning on online feedback. It can handle volatile arms by incorporating agent cards as defined by the Agent2Agent Protocol without requiring changes to the internal structures or retraining. Our experimental evaluation shows that CoCoMaMa and Neural-CoCoMaMa achieve better results than respective state-of-the-art algorithms using the LLM routing dataset SPROUT and a novel extended version of SPROUT with synthetic specialized agents.

## Installation

1. **Create and activate a virtual environment:**
```bash
# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. **Install the required packages:**
```bash
pip install -r requirements.txt
```

3. **Set up Hugging Face access:**
   - Create a Hugging Face account at https://huggingface.co/
   - Login using the Hugging Face CLI:
   ```bash
   huggingface-cli login
   ```
   - This is required to access the SPROUT dataset used in the benchmarks.

## Usage

### Quick Start

**Interactive Mode:**
```bash
python main_streaming.py
```
This will guide you through the configuration process and automatically save your settings.

**Paper Reproduction:**
```bash
# SPROUT experiments
python main_streaming.py --config_file config_sprout_paper.yaml

# Synthetic experiments  
python main_streaming.py --config_file config_synthetic_paper.yaml

# Ablation studies
python main_streaming.py --ablation_config ablation_config_synthetic_theta.yaml
```

### Available Algorithms

- **CoCoMaMa**: Combinatorial Contextual Multi-Armed Bandit with statistically informed splits
- **Neural-CoCoMaMa**: Neural network-enhanced version of CoCoMaMa
- **HD-ACC-UCB**: High-Dimensional Contextual Combinatorial Volatile Multi-armed Bandit with Adaptive Discretization using Hyperrectangles instead of Hypercubes
- **CC-MAB**: Contextual Combinatorial Multi-Armed Bandit
- **Neural-MAB**: Neural Multi-Armed Bandit
- **Streaming Versions**: Modified versions of all algorithms for streaming data scenarios
- **Random**: Random agent selection (baseline)
- **Benchmark**: Oracle benchmark (selects best agents)

## Results

The experiments generate several output files:

- **Results**: Detailed performance data in CSV format along with generated plots
- **Configs**: Saved configuration files for reproducibility if using the interactive mode

Results are saved in the `results/` directory with timestamps for easy identification.

## Dataset Attribution

This project uses the SPROUT dataset from:
- **Paper**: [CARROT: A Dataset for LLM Routing](https://arxiv.org/pdf/2502.03261)
- **Dataset**: [CARROT-LLM-Routing/SPROUT](https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT)

## Algorithm Attributions

### Original Implementations
- **HD-ACC-UCB**: Based on the implementation from [Bilkent-CYBORG/ACC-UCB](https://github.com/Bilkent-CYBORG/ACC-UCB) by Nika, Andi, Sepehr Elahi, and Cem Tekin, modified to use Hyperrectangles instead of Hypercubes
- **CC-MAB**: Based on the implementation from [Bilkent-CYBORG/ACC-UCB](https://github.com/Bilkent-CYBORG/ACC-UCB) by Nika, Andi, Sepehr Elahi, and Cem Tekin. Algorithm originally proposed by Chen, Lixing, Jie Xu, and Zhuo Lu.


### Original Implementations by Jonathan Rau
- **CoCoMaMa**: Combinatorial Contextual Multi-Armed Bandit with statistically informed splits
- **Neural-CoCoMaMa**: Neural network-enhanced version of CoCoMaMa
- **Neural-MAB**: Implementation based on existing methodology from Lin, Shouxu, et al.

### Streaming Versions by Jonathan Rau
- **Streaming HD-ACC-UCB**: Modified by Jonathan Rau for streaming data scenarios, uses Hyperrectangles for high-dimensional contexts
- **Streaming CC-MAB**: Modified by Jonathan Rau for streaming data scenarios
- **Streaming CoCoMaMa**: Modified by Jonathan Rau for streaming data scenarios
- **Streaming Neural-CoCoMaMa**: Modified by Jonathan Rau for streaming data scenarios
- **Streaming Neural-MAB**: Modified by Jonathan Rau for streaming data scenarios

### Baseline Algorithms
- **Random**: Random agent selection (baseline)
- **Benchmark**: Oracle benchmark (selects best agents)

## Citation

If you use this code in your research, please cite: Rau, Jonathan, et al. "CoCoMaMa: Contextual Combinatorial Multi-Armed Bandit Router for Multi-Agent Systems with Volatile Arms", to appear in The Second International Workshop on Hypermedia Multi-Agent Systems
(HyperAgents 2025), in conjunction with the 28th European Conference on
Artificial Intelligence (ECAI 2025).

### Additional Citations

For the original ACC-UCB, please also cite: Nika, Andi, Sepehr Elahi, and Cem Tekin. "Contextual combinatorial volatile multi-armed bandit with adaptive discretization." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

For the CC-MAB algorithm, please also cite: Chen, Lixing, Jie Xu, and Zhuo Lu. "Contextual combinatorial multi-armed bandits with volatile arms and submodular reward." Advances in Neural Information Processing Systems 31 (2018).

For the Neural-MAB algorithm, please also cite: Lin, Shouxu, et al. "A neural-based bandit approach to mobile crowdsourcing." Proceedings of the 23rd Annual International Workshop on Mobile Computing Systems and Applications. 2022.