import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Discrete
from torch.distributions import Categorical, MultivariateNormal
from tqdm import tqdm

from ..utils.common import TrainingResult


class Network(nn.Module):
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


class RolloutBuffer(object):
    def __init__(self, T, N, dim_S, dim_A, discrete, device):
        self.T, self.N = T, N
        self.discrete = discrete
        self.device = device

        self.states = torch.zeros((T, N, dim_S), device=device)
        self.actions = (
            torch.zeros((T, N), dtype=torch.long, device=device)
            if discrete else torch.zeros((T, N, dim_A), device=device)
        )
        self.logprobs = torch.zeros((T, N), device=device)
        self.rewards = torch.zeros((T, N), device=device)
        self.values = torch.zeros((T, N), device=device)
        self.dones = torch.zeros((T, N), device=device)
        self.advantages = torch.zeros((T, N), device=device)
        self.returns = torch.zeros((T, N), device=device)

    def store(self, t, state, action, logprob, reward, value, done):
        self.states[t] = state
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.rewards[t] = reward
        self.values[t] = value
        self.dones[t] = done

    def compute_gae(self, last_value, gamma, lam):
        adv = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self.values[t + 1]
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            adv = delta + gamma * lam * mask * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def iter_minibatches(self, batch_size):
        flat_states = self.states.reshape(-1, self.states.shape[-1])
        flat_actions = (
            self.actions.reshape(-1)
            if self.discrete else self.actions.reshape(-1, self.actions.shape[-1])
        )
        flat_logprobs = self.logprobs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)

        n_total = flat_states.shape[0]
        idxs = torch.randperm(n_total, device=self.device)
        for start in range(0, n_total, batch_size):
            mb = idxs[start:start + batch_size]
            yield (
                flat_states[mb],
                flat_actions[mb],
                flat_logprobs[mb],
                flat_advantages[mb],
                flat_returns[mb],
            )


class RunPPO(object):
    def __init__(self,
                 env,
                 T=128,
                 K=10,
                 batch_size=64,
                 lr=3e-4,
                 gamma=0.99,
                 lam=0.95,
                 clip_eps=0.2,
                 c1=0.5,
                 c2=0.01,
                 max_grad_norm=0.5,
                 arch=None,
                 action_std_init=0.6,
                 device=None):

        self.env = env
        self.N = env.num_envs
        self.T = T
        self.K = K
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.dim_S = env.single_observation_space.shape[0]
        self.discrete = isinstance(env.single_action_space, Discrete)
        self.dim_A = env.single_action_space.n if self.discrete else env.single_action_space.shape[0]

        hidden = arch if arch is not None else [64, 64]
        self.actor = Network([self.dim_S, *hidden, self.dim_A]).to(self.device)
        self.critic = Network([self.dim_S, *hidden, 1]).to(self.device)

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        if not self.discrete:
            self.log_std = nn.Parameter(
                torch.full((self.dim_A,), float(np.log(action_std_init)), device=self.device)
            )
            params.append(self.log_std)

        self.optimizer = optim.Adam(params, lr=lr)
        self.buffer = RolloutBuffer(T, self.N, self.dim_S, self.dim_A, self.discrete, self.device)
        self.total_env_steps = 0

    def _policy(self, state):
        out = self.actor(state)
        if self.discrete:
            return Categorical(logits=out)

        std = self.log_std.exp().expand_as(out)
        cov = torch.diag_embed(std ** 2)
        return MultivariateNormal(out, cov)

    @torch.no_grad()
    def _act(self, state):
        dist = self._policy(state)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.critic(state).squeeze(-1)
        return action, logprob, value

    def _collect_rollout(self, state, ep_rewards, episode_steps, running_reward):
        for t in range(self.T):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, logprob, value = self._act(state_t)

            action_np = action.cpu().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = np.logical_or(terminated, truncated)
            self.total_env_steps += self.N

            self.buffer.store(
                t,
                state_t,
                action,
                logprob,
                torch.as_tensor(reward, dtype=torch.float32, device=self.device),
                value,
                torch.as_tensor(done, dtype=torch.float32, device=self.device),
            )

            running_reward += reward
            for i, is_done in enumerate(done):
                if is_done:
                    ep_rewards.append(float(running_reward[i]))
                    episode_steps.append(self.total_env_steps)
                    running_reward[i] = 0.0

            state = next_state

        with torch.no_grad():
            last_value = self.critic(
                torch.as_tensor(state, dtype=torch.float32, device=self.device)
            ).squeeze(-1)
        self.buffer.compute_gae(last_value, self.gamma, self.lam)
        return state

    def _update(self):
        losses = []
        for _ in range(self.K):
            for states, actions, old_logprobs, advantages, returns in self.buffer.iter_minibatches(self.batch_size):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                dist = self._policy(states)
                new_logprobs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logprobs - old_logprobs)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(states).squeeze(-1)
                value_loss = (values - returns).pow(2).mean()
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.max_grad_norm)
                self.optimizer.step()

                losses.append({
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item(),
                })
        return losses

    def learn(self, total_iterations, progress=True):
        state, _ = self.env.reset()
        running_reward = np.zeros(self.N, dtype=np.float32)
        ep_rewards = []
        episode_steps = []
        losses = []

        for _ in tqdm(range(total_iterations), disable=not progress):
            state = self._collect_rollout(state, ep_rewards, episode_steps, running_reward)
            losses.extend(self._update())

        return TrainingResult(ep_rewards, episode_steps, losses)
