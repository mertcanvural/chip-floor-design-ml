24Fall Machine Learning Term Project
Studied Paper
Title: MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning
Authors: Lai, Yao, Yao Mu, and Ping Luo
Conference: Advances in Neural Information Processing Systems 35 (2022) - NeurIPS 2022, Spotlight

Paper Link: MaskPlace Paper (arXiv)
Original GitHub Repository: MaskPlace Repository

Project Overview
This project aims to reproduce and build upon the results of the MaskPlace methodology. MaskPlace uses reinforcement learning (RL) with Proximal Policy Optimization (PPO) for chip placement optimization, focusing on reducing wirelength and ensuring zero-overlap placements. Our objective was to validate the MaskPlace results and explore potential improvements through advanced metrics tracking and visualizations.

Teammates and Contributions

1. Mert Can Vural (GitHub: mertcanvural)
Initial Reward Graphs: Generated the early reward progression graphs to visualize the cumulative rewards during training.
Graph Contributions:
Reward Progression Plot
Action Loss Plot
Value Loss Plot

Code Implementation: Reproduced the original MaskPlace codebase and set up the environment on Google Colab.
Analysis: Contributed to debugging issues related to the PPO agent and validating the training process.

2. Taha Oğuzhan Uçar (GitHub: ozi14)
Detailed Graph Analysis: Produced detailed visualizations and analysis, including key metrics for wirelength and cost.
Graph Contributions:
Cost and HPWL Metrics Plot
Reward Metrics During Training Plot
Training Scores Over Epochs Plot
Final Placement Layout Visualizations
Further Analysis: Conducted in-depth analysis of the model's convergence behavior and comparison with baseline methods.
Documentation: Assisted with documenting the experimental setup and analysis sections.
