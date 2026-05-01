## Possible Citations:

- Fujimoto, S., Hoof, H., & Meger, D. (2018). *Addressing function approximation error in actor-critic methods*. Proceedings of the 35th International Conference on Machine Learning (ICML). https://arxiv.org/abs/1802.09477

- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2015). *Continuous control with deep reinforcement learning*. arXiv preprint arXiv:1509.02971. https://arxiv.org/abs/1509.02971

- Hasselt, H. V. (2010). *Double Q-learning*. Advances in Neural Information Processing Systems. https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html

## Ablation Works that can be used in the Report:

> Twin critics vs one critic. Removing `Q2` and the `min(Q1, Q2)` target turns TD3 closer to DDPG and should expose overestimation bias.

> Delayed policy updates. Sweep `policy_delay` in {1, 2, 4}. `policy_delay=1` removes the delayed update idea, while larger values make the critic move further between actor updates.

> Target policy smoothing. Set `target_noise=0` or vary `noise_clip` to measure how smoothing affects critic sharpness and stability.

> Exploration noise. Sweep `act_noise` in {0.05, 0.1, 0.2, 0.4}. Too little noise can under-explore; too much noise makes the deterministic actor behave like a noisy random policy early on.

> Start-step random exploration. Sweep `start_steps` in {0, 500, 1000, 5000}. This checks how much TD3 depends on a diverse initial replay buffer.

> Polyak coefficient sensitivity. `polyak=0.995` is standard, but lower values update target networks faster and can destabilize targets or improve adaptation depending on the environment.

> TD3 vs SAC on the same continuous-control task. Both use replay buffers, twin critics, and Polyak targets, but TD3 uses a deterministic actor with external noise while SAC uses a stochastic entropy-regularized actor.
