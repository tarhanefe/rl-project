## Possible Citations:

- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor*. Proceedings of the 35th International Conference on Machine Learning (ICML). https://arxiv.org/abs/1801.01290

- Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., & Levine, S. (2018). *Soft actor-critic algorithms and applications*. arXiv preprint arXiv:1812.05905. https://arxiv.org/abs/1812.05905

- Fujimoto, S., Hoof, H., & Meger, D. (2018). *Addressing function approximation error in actor-critic methods*. Proceedings of the 35th International Conference on Machine Learning (ICML). https://arxiv.org/abs/1802.09477

- OpenAI Spinning Up. *Soft Actor-Critic*. https://spinningup.openai.com/en/latest/algorithms/sac.html

## Ablation Works that can be used in the Report:

> Fixed `alpha` vs automatic entropy tuning. The notebook follows Spinning Up's simpler fixed-temperature SAC, but the later SAC version learns `alpha` to target a desired entropy. This is one of the highest-value ablations because `alpha` controls the exploration/exploitation trade-off directly.

> Twin critics vs one critic. Removing `Q2` and the `min(Q1, Q2)` target should expose overestimation bias, similar to the TD3 motivation. Compare learning stability and final Pendulum return.

> Effect of `start_steps`. SAC often benefits from random actions before policy sampling starts. Sweep `start_steps` in {0, 500, 1000, 5000} and compare early learning speed and final performance.

> Polyak coefficient sensitivity. `polyak=0.995` is standard, but using smaller values updates targets faster. This can improve responsiveness or destabilize Q targets depending on environment and learning rate.

> State-dependent log standard deviation vs state-independent log standard deviation. Spinning Up notes that SAC needs the policy standard deviation to depend on state. Replacing `log_std_layer` with one learned vector is a clean implementation ablation.

> Batch size and update ratio. Spinning Up keeps the environment-step to gradient-step ratio near 1 by doing `update_every` updates every `update_every` steps. Try changing that ratio to see when replay reuse helps or causes overfitting to stale data.
