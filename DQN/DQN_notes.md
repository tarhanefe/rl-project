## Possible Citations: 

- Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). *Playing Atari with deep reinforcement learning*. arXiv preprint arXiv:1312.5602. https://arxiv.org/abs/1312.5602

- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533. https://doi.org/10.1038/nature14236

- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2015). *Continuous control with deep reinforcement learning*. arXiv preprint arXiv:1509.02971. https://arxiv.org/abs/1509.02971

## Ablation Works that can be used in the Report: 

> What is the effect of using only a single network for Q updates vs training another network as the target Q function. In the original ATARI paper it was a single network whereas in the second paper published in nature after 2 years contains a single function. We can check that 

> What is the effect of using soft or hard updates on the target network using the actual Q network. The 2015 paper uses hard updates whereas the 3rd paper uses soft updates with EMA like modal averaging. We can see its effects. 

> There are several params like eps, clipping, starting step, update step, etc. etc. We can do ablations on them but wouldn't prefer at first. We can also play with the model parameters.