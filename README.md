# Impartial Games: A Challenge for Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-Machine%20Learning%202026-blue)](https://link.springer.com/article/10.1007/s10994-026-06996-1)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official code repository for the paper:

> **Impartial Games: A Challenge for Reinforcement Learning**  
> Bei Zhou, Søren Riis  
> *Machine Learning*, vol. 115, no. 3, p. 54, 2026, Springer  
> [[Paper]](https://link.springer.com/article/10.1007/s10994-026-06996-1) · [[arXiv]](https://arxiv.org/abs/2205.12787)

---

## Overview

AlphaZero-style reinforcement learning algorithms have achieved superhuman performance in partisan games such as Chess, Shogi, and Go. This paper demonstrates that these same algorithms encounter **fundamental and systematic challenges** when applied to *impartial games* — a class of combinatorial games where both players share the same moves, and optimal strategy requires implicitly learning a parity function.

We use **Nim** as a concrete and mathematically tractable case study. Our key contributions are:

- A novel **champion vs. expert** mastery framework for evaluating RL agent performance in combinatorial games.
- Empirical evidence that AlphaZero-style agents can achieve champion-level play on small Nim boards but fail to scale to expert-level play as board size increases.
- Analysis of the underlying structural reasons — rooted in parity and the Sprague-Grundy theorem — why self-play RL struggles with impartial games.
- Board-level move analysis revealing the specific failure modes of trained agents.

---

## Repository Structure

```
.
├── example/                 # Example 2 from Section 4: Two Levels of Mastery
├── reinforcement learning/  # AlphaZero-style RL experiments on Nim (Section 5)
├── deep_RL/                 # Deep RL experiments and network training scripts
├── tabular_RL/              # Tabular RL baselines (Q-learning, value iteration)
├── partisan_nim/            # Experiments on partisan variants of Nim
├── multi_frame_nim/         # Multi-frame state representation experiments
├── data/                    # All data generated and used in the paper
├── images/                  # Figures used in this README and the paper
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sagebei/Impartial-games-RL.git
cd Impartial-games-RL
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate    
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

Refer to the paper for full experiment configurations. Each experiment folder contains its own scripts; the main configuration is controlled via the `args` dictionary at the top of each script.

### Key configuration options

| Parameter | Description | Default |
|---|---|---|
| `calculate_elo` | Compute Elo ratings against ancestor agents. Expensive — disable if not needed. | `true` |
| `num_workers` | Number of parallel Ray workers. Should not exceed your CPU count. | `4` |
| `board_size` | Number of Nim heaps (5, 6, or 7 supported for analysis). | `5` |

**Disabling Elo calculation** (recommended unless you need relative strength comparisons):

```python
args = {
    'calculate_elo': False,
    ...
}
```

**Setting the number of parallel workers:**

We use [Ray](https://www.ray.io/) for parallelism. Set `num_workers` to at most the number of CPUs available on your machine:

```python
args = {
    'num_workers': 4,   # adjust to your hardware
    ...
}
```

---

## Trained Models

Pre-trained policy and value network checkpoints are stored in the `model/` folder. The filename reflects the Nim board configuration used for training:

| File | Description |
|---|---|
| `5heaps` | Policy/value network trained on 5-heap Nim |
| `6heaps` | Policy/value network trained on 6-heap Nim |
| `7heaps` | Policy/value network trained on 7-heap Nim |

Position-level analysis is supported for 5, 6, and 7-heap Nim using these models.

## Hardware and Runtime

Training times reported in the paper are based on an **NVIDIA A100 GPU** on the [QMUL High Performance Computing cluster](https://www.qmul.ac.uk/its/research/hpc/). Specific benchmarks:

- **7-heap Nim (AlphaZero training):** 8 CPUs + 1 GPU
- **5- and 6-heap Nim:** Single GPU sufficient

Your actual runtime will vary depending on available hardware. The paper's specifications serve as a useful reference for estimation.

---

## Related Work

This repository also contains code related to the follow-up theoretical work:

> **Mastering NIM and Impartial Games with Weak Neural Networks: An AlphaZero-inspired Multi-Frame Approach**  
> Søren Riis  
> *arXiv preprint*, 2024 · [[arXiv]](https://arxiv.org/abs/2411.06403)

The `multi_frame_nim/` folder contains experiments exploring multi-frame state representations, which provably overcome the parity barrier identified in the main paper.

---

## Citation

If this code or paper contributes to your research, please cite:

```bibtex
@article{zhou2026impartial,
  title={Impartial Games: A Challenge for Reinforcement Learning},
  author={Zhou, Bei and Riis, Soren},
  journal={Machine Learning},
  volume={115},
  number={3},
  pages={54},
  year={2026},
  publisher={Springer US New York}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
