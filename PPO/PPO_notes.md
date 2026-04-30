## Possible Citations:

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy optimization algorithms*. arXiv preprint arXiv:1707.06347. https://arxiv.org/abs/1707.06347

- Schulman, J., Levine, S., Abbeel, P., Jordan, M. I., & Moritz, P. (2015). *Trust region policy optimization*. Proceedings of the 32nd International Conference on Machine Learning (ICML), 1889–1897. https://arxiv.org/abs/1502.05477

- Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2016). *High-dimensional continuous control using generalized advantage estimation*. International Conference on Learning Representations (ICLR). https://arxiv.org/abs/1506.02438

- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). *Asynchronous methods for deep reinforcement learning*. Proceedings of the 33rd International Conference on Machine Learning (ICML), 1928–1937. https://arxiv.org/abs/1602.01783

- Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). *Implementation matters in deep policy gradients: A case study on PPO and TRPO*. International Conference on Learning Representations (ICLR). https://arxiv.org/abs/2005.12729

## Ablation Works that can be used in the Report:

> The PPO paper proposes two surrogate losses: the clipped objective `L^CLIP` (Eq. 7) and an adaptive KL-penalty objective `L^KLPEN` (Eq. 8). The paper itself shows the clipped one wins, but reproducing that ablation on a small env like CartPole or Pendulum is a clean exercise.

> The clip range ε is the single most prominent hyperparameter of PPO. Sweeping ε ∈ {0.05, 0.1, 0.2, 0.3, 0.5} shows the trade-off between conservative updates (small ε, slow but stable) and aggressive updates (large ε, faster but can collapse).

> GAE has two knobs: γ (already in the algorithm) and λ. λ=0 is one-step TD, λ=1 is Monte-Carlo. Most reference implementations use 0.95 — checking sensitivity to this choice on Pendulum is a quick ablation.

> The N actors × T timesteps trade-off. The paper uses N·T ∈ {2048, 4096} for MuJoCo. We can fix the total `N·T` and vary the split between N and T — does parallelism help, or only the total batch size?

> K (the number of update epochs per rollout) controls how aggressively we re-use data. K=1 reduces PPO to one-step actor-critic. K=10 is standard. Higher K relies more heavily on the clipping to prevent destructive updates — a clean way to see the clip's role.

> The Engstrom et al. paper (cited above) shows that the **engineering tricks** (observation normalization, reward normalization, advantage normalization, learning-rate annealing, value clipping, orthogonal init) often matter as much as the core algorithm. The notebook already demonstrates the **observation normalization** ablation on Pendulum (Section 6); the rest can be added one by one.

> Shared vs separate actor–critic networks. Paper §6.1 uses separate. With shared parameters the value loss can dominate or starve the policy gradient, which is why a coefficient `c1` exists at all. We can compare both.

> Discrete (CartPole) vs continuous (Pendulum) is a built-in ablation already, since the same `RunPPO` class handles both via the `_policy` branch on `Discrete` action spaces. Worth highlighting in the report that one algorithm covers both regimes — a key selling point of PPO over DQN/DDPG.
