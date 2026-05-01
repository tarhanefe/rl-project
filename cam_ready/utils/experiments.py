import gymnasium as gym
import numpy as np

from ..src.dqn import RunDQN
from ..src.ppo import RunPPO
from ..src.sac import RunSAC
from ..src.td3 import RunTD3
from .common import seed_everything


def make_env(env_id, seed):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def make_vector_env(env_id, n_actors, seed):
    env = gym.vector.SyncVectorEnv(
        [lambda env_id=env_id: gym.make(env_id) for _ in range(n_actors)]
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def train_dqn_cartpole(seed, total_env_steps=50_000, progress=True, agent_overrides=None):
    seed_everything(seed)
    env = make_env("CartPole-v1", seed)
    try:
        config = dict(
            lr=5e-4,
            batch_size=256,
            gamma=0.99,
            arch=[env.observation_space.shape[0], 128, 128, env.action_space.n],
            clip=1.0,
            eps=(0.95, 600, 0.05),
            step_skipping=2,
            tau=0.01,
            target_update_freq=0,
            start_step=100,
            memory_size=100_000,
        )
        if agent_overrides:
            config.update(agent_overrides)
        agent = RunDQN(env=env, **config)
        return agent.learn(total_steps=total_env_steps, max_steps=500, progress=progress)
    finally:
        env.close()


def train_ppo_cartpole(seed, total_env_steps=50_000, n_actors=4, T=128, progress=True, agent_overrides=None):
    seed_everything(seed)
    env = make_vector_env("CartPole-v1", n_actors=n_actors, seed=seed)
    total_iterations = max(1, int(total_env_steps // (T * n_actors)))
    try:
        config = dict(
            T=T,
            K=4,
            batch_size=64,
            lr=3e-4,
            gamma=0.99,
            lam=0.95,
            clip_eps=0.2,
            c1=0.5,
            c2=0.01,
            arch=[64, 64],
        )
        if agent_overrides:
            config.update(agent_overrides)
        agent = RunPPO(env=env, **config)
        return agent.learn(total_iterations=total_iterations, progress=progress)
    finally:
        env.close()


def train_ppo_pendulum(seed, total_env_steps=25_000, n_actors=4, T=256, progress=True, agent_overrides=None):
    seed_everything(seed)
    env = make_vector_env("Pendulum-v1", n_actors=n_actors, seed=seed)
    total_iterations = max(1, int(total_env_steps // (T * n_actors)))
    try:
        config = dict(
            T=T,
            K=5,
            batch_size=64,
            lr=3e-4,
            gamma=0.99,
            lam=0.95,
            clip_eps=0.2,
            c1=0.5,
            c2=0.0,
            arch=[64, 64],
            action_std_init=0.6,
        )
        if agent_overrides:
            config.update(agent_overrides)
        agent = RunPPO(env=env, **config)
        return agent.learn(total_iterations=total_iterations, progress=progress)
    finally:
        env.close()


def train_sac_pendulum(seed, total_env_steps=25_000, progress=True, agent_overrides=None):
    seed_everything(seed)
    env = make_env("Pendulum-v1", seed)
    try:
        config = dict(
            lr=1e-3,
            batch_size=100,
            gamma=0.99,
            polyak=0.995,
            alpha=0.2,
            hidden_sizes=(256, 256),
            replay_size=100_000,
            start_steps=1_000,
            update_after=1_000,
            update_every=50,
            max_ep_len=200,
        )
        if agent_overrides:
            config.update(agent_overrides)
        agent = RunSAC(env=env, **config)
        return agent.learn(
            total_steps=total_env_steps,
            eval_every=5_000,
            num_test_episodes=5,
            progress=progress,
        )
    finally:
        env.close()


def train_td3_pendulum(seed, total_env_steps=25_000, progress=True, agent_overrides=None):
    seed_everything(seed)
    env = make_env("Pendulum-v1", seed)
    try:
        config = dict(
            pi_lr=1e-3,
            q_lr=1e-3,
            batch_size=100,
            gamma=0.99,
            polyak=0.995,
            hidden_sizes=(256, 256),
            replay_size=100_000,
            start_steps=1_000,
            update_after=1_000,
            update_every=50,
            policy_delay=2,
            act_noise=0.1,
            target_noise=0.2,
            noise_clip=0.5,
            max_ep_len=200,
        )
        if agent_overrides:
            config.update(agent_overrides)
        agent = RunTD3(env=env, **config)
        return agent.learn(
            total_steps=total_env_steps,
            eval_every=5_000,
            num_test_episodes=5,
            progress=progress,
        )
    finally:
        env.close()


def run_discrete_cartpole_comparison(seeds=(0, 16, 25), total_env_steps=50_000, progress=True):
    results = {"DQN": [], "PPO": []}
    for seed in seeds:
        results["DQN"].append(train_dqn_cartpole(seed, total_env_steps, progress=progress))
        results["PPO"].append(train_ppo_cartpole(seed, total_env_steps, progress=progress))
    return results


def run_continuous_pendulum_comparison(seeds=(0, 16, 25), total_env_steps=25_000, progress=True):
    results = {"PPO": [], "SAC": [], "TD3": []}
    for seed in seeds:
        results["PPO"].append(train_ppo_pendulum(seed, total_env_steps, progress=progress))
        results["SAC"].append(train_sac_pendulum(seed, total_env_steps, progress=progress))
        results["TD3"].append(train_td3_pendulum(seed, total_env_steps, progress=progress))
    return results


def run_algorithm_ablations(train_fn, variants, seeds=(0,), total_env_steps=10_000, progress=True):
    results = {}
    for variant_name, overrides in variants.items():
        results[variant_name] = []
        for seed in seeds:
            results[variant_name].append(
                train_fn(
                    seed=seed,
                    total_env_steps=total_env_steps,
                    progress=progress,
                    agent_overrides=overrides,
                )
            )
    return results


DQN_CARTPOLE_ABLATIONS = {
    "baseline": {},
    "hard_target_100": {"target_update_freq": 100},
    "fast_eps_decay": {"eps": (0.95, 200, 0.05)},
    "small_replay": {"memory_size": 5_000},
}

PPO_CARTPOLE_ABLATIONS = {
    "baseline": {},
    "clip_0.10": {"clip_eps": 0.10},
    "clip_0.30": {"clip_eps": 0.30},
    "no_entropy_bonus": {"c2": 0.0},
}

PPO_PENDULUM_ABLATIONS = {
    "baseline": {},
    "clip_0.10": {"clip_eps": 0.10},
    "lambda_0.80": {"lam": 0.80},
    "lower_action_std": {"action_std_init": 0.30},
}

SAC_PENDULUM_ABLATIONS = {
    "baseline": {},
    "alpha_0.05": {"alpha": 0.05},
    "alpha_0.50": {"alpha": 0.50},
    "no_random_warmup": {"start_steps": 0},
}

TD3_PENDULUM_ABLATIONS = {
    "baseline": {},
    "no_target_smoothing": {"target_noise": 0.0},
    "policy_delay_1": {"policy_delay": 1},
    "act_noise_0.20": {"act_noise": 0.20},
}


def run_dqn_cartpole_ablations(seeds=(0,), total_env_steps=10_000, progress=True):
    return run_algorithm_ablations(
        train_dqn_cartpole,
        DQN_CARTPOLE_ABLATIONS,
        seeds=seeds,
        total_env_steps=total_env_steps,
        progress=progress,
    )


def run_ppo_cartpole_ablations(seeds=(0,), total_env_steps=10_000, progress=True):
    return run_algorithm_ablations(
        train_ppo_cartpole,
        PPO_CARTPOLE_ABLATIONS,
        seeds=seeds,
        total_env_steps=total_env_steps,
        progress=progress,
    )


def run_ppo_pendulum_ablations(seeds=(0,), total_env_steps=10_000, progress=True):
    return run_algorithm_ablations(
        train_ppo_pendulum,
        PPO_PENDULUM_ABLATIONS,
        seeds=seeds,
        total_env_steps=total_env_steps,
        progress=progress,
    )


def run_sac_pendulum_ablations(seeds=(0,), total_env_steps=10_000, progress=True):
    return run_algorithm_ablations(
        train_sac_pendulum,
        SAC_PENDULUM_ABLATIONS,
        seeds=seeds,
        total_env_steps=total_env_steps,
        progress=progress,
    )


def run_td3_pendulum_ablations(seeds=(0,), total_env_steps=10_000, progress=True):
    return run_algorithm_ablations(
        train_td3_pendulum,
        TD3_PENDULUM_ABLATIONS,
        seeds=seeds,
        total_env_steps=total_env_steps,
        progress=progress,
    )


def summarize_final_scores(results_by_algorithm, window=10):
    rows = []
    for algorithm, results in results_by_algorithm.items():
        scores = []
        for result in results:
            rewards = np.asarray(result.episode_rewards, dtype=np.float32)
            if len(rewards) == 0:
                scores.append(np.nan)
            else:
                scores.append(float(rewards[-min(window, len(rewards)):].mean()))
        rows.append({
            "algorithm": algorithm,
            "mean_final_return": float(np.nanmean(scores)),
            "std_final_return": float(np.nanstd(scores)),
            "per_seed": scores,
        })
    return rows
