import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..utils.common import TrainingResult


class ReplayMemory(object):
    def __init__(self, memory_size, batch_size):
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = []

    def insert(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states = [item[0] for item in batch]
        actions = [item[1] for item in batch]
        rewards = [item[2] for item in batch]
        next_states = [item[3] for item in batch]
        dones = [item[4] for item in batch]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, arch):
        super().__init__()
        if len(arch) < 2:
            raise ValueError("Layer size cannot be smaller than 2")

        layers = []
        for in_features, out_features in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(in_features, out_features))
            if out_features != arch[-1]:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RunDQN(object):
    def __init__(self,
                 env,
                 lr=5e-4,
                 batch_size=256,
                 gamma=0.99,
                 arch=None,
                 clip=1.0,
                 eps=(0.9, 2500, 0.1),
                 step_skipping=4,
                 tau=0.005,
                 target_update_freq=0,
                 start_step=100,
                 memory_size=10000,
                 device=None):

        self.step = 0
        self.env = env
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip = clip
        self.eps = eps
        self.step_skipping = step_skipping
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.memory_size = memory_size
        self.start_step = start_step
        self.dim_A = env.action_space.n
        self.dim_S = env.observation_space.shape[0]
        self.arch = arch if arch is not None else [self.dim_S, 128, 128, self.dim_A]

        if self.arch[0] != self.dim_S or self.arch[-1] != self.dim_A:
            raise ValueError("arch must start with the state size and end with the action size")
        if self.target_update_freq < 0:
            raise ValueError("target_update_freq must be non-negative")
        if self.target_update_freq == 0 and (self.tau is None or self.tau <= 0):
            raise ValueError("tau must be positive when target_update_freq is 0")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.replay_buffer = ReplayMemory(self.memory_size, self.batch_size)

        self.Q_network = QNetwork(self.arch).to(self.device)
        self.Q_target = QNetwork(self.arch).to(self.device)
        self.Q_target.load_state_dict(self.Q_network.state_dict())
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.lr)

        self.env_step = 0
        self.learn_step = 0

    def _reset_env(self, seed=None):
        reset_result = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        return reset_result[0] if isinstance(reset_result, tuple) else reset_result

    def _step_env(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result
        return next_state, reward, done

    def choose_action(self, state):
        eps_start, decay_steps, eps_end = self.eps
        self.step_eps = eps_end + (eps_start - eps_end) * np.exp(-self.step / decay_steps)
        self.step += 1

        if random.random() < self.step_eps:
            return random.randrange(self.dim_A)

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_values = self.Q_network(state_tensor)
            return action_values.argmax(dim=1).item()

    def hard_update(self):
        self.Q_target.load_state_dict(self.Q_network.state_dict())

    def soft_update(self):
        for target_param, online_param in zip(self.Q_target.parameters(), self.Q_network.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.Q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.Q_target(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.Q_network.parameters(), self.clip)
        self.optimizer.step()

        return loss.item()

    def learn(self, num_episodes=None, max_steps=500, total_steps=None, progress=True):
        if num_episodes is None and total_steps is None:
            raise ValueError("Pass num_episodes, total_steps, or both")

        episode_rewards = []
        episode_steps = []
        losses = []

        step_progress = num_episodes is None and total_steps is not None
        progress_total = total_steps if step_progress else num_episodes
        iterator = tqdm(total=progress_total, disable=not progress) if progress_total is not None else None
        try:
            while True:
                if num_episodes is not None and len(episode_rewards) >= num_episodes:
                    break
                if total_steps is not None and self.env_step >= total_steps:
                    break

                state = self._reset_env()
                total_r = 0.0

                for _ in range(max_steps):
                    if total_steps is not None and self.env_step >= total_steps:
                        break

                    action = self.choose_action(state)
                    next_state, reward, done = self._step_env(action)

                    self.replay_buffer.insert(state, action, reward, next_state, done)
                    state = next_state
                    total_r += reward
                    self.env_step += 1
                    if step_progress and iterator is not None:
                        iterator.update(1)

                    if self.env_step > self.start_step and self.env_step % self.step_skipping == 0:
                        loss = self.train()
                        if loss is not None:
                            losses.append(loss)
                            self.learn_step += 1

                            if self.target_update_freq == 0:
                                self.soft_update()
                            elif self.learn_step % self.target_update_freq == 0:
                                self.hard_update()

                    if done:
                        break

                episode_rewards.append(float(total_r))
                episode_steps.append(self.env_step)
                if not step_progress and iterator is not None:
                    iterator.update(1)
        finally:
            if iterator is not None:
                iterator.close()

        return TrainingResult(episode_rewards, episode_steps, losses)
