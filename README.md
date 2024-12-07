# 24Fall Machine Learning Term Project

## Studied Paper

**Title:** [MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning](https://arxiv.org/abs/2210.07805)  
**Authors:** Lai, Yao, Yao Mu, and Ping Luo  
**Conference:** *Advances in Neural Information Processing Systems 35 (2022) - NeurIPS 2022, Spotlight*  

- **[Original GitHub Repository: MaskPlace Repository](https://github.com/MaskPlace/MaskPlace)**  

---

## Project Overview

This project aims to reproduce and build upon the results of the *MaskPlace* methodology. MaskPlace leverages reinforcement learning (RL) with Proximal Policy Optimization (PPO) for chip placement optimization, focusing on reducing wirelength and ensuring zero-overlap placements. Our primary goals were:

1. Validate the original MaskPlace results.
2. Explore potential improvements through advanced metrics tracking and visualizations.

---

## Teammates and Contributions

### 1. Mert Can Vural ([GitHub: mertcanvural](https://github.com/mertcanvural))

#### Contributions:

- **Initial Reward Graphs:** Generated early reward progression graphs to visualize cumulative rewards during training.

  **Graph Contributions:**
  - Reward Progression Plot
  - Action Loss Plot
  - Value Loss Plot

- **Code Implementation:**  
  - Reproduced the original MaskPlace codebase.  
  - Set up the environment on Google Colab.

- **Analysis:**  
  - Debugged PPO agent issues.  
  - Validated the training process.

### 2. Taha Oğuzhan Uçar ([GitHub: ozi14](https://github.com/ozi14))

#### Contributions:

- **Detailed Graph Analysis:** Produced visualizations and analysis of key metrics (wirelength, cost, etc.).

  **Graph Contributions:**
  - Cost and HPWL Metrics Plot
  - Reward Metrics During Training Plot
  - Training Scores Over Epochs Plot
  - Final Placement Layout Visualizations

- **Further Analysis:**  
  - Analyzed model convergence behavior.  
  - Compared results with baseline methods.

- **Documentation:**  
  - Assisted with documenting the experimental setup and analysis sections.

---
## License

This project follows the same license as the original MaskPlace repository. Please refer to the [original license](https://github.com/MaskPlace/MaskPlace/blob/main/LICENSE) for more information.

---
