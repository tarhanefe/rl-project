from dataclasses import dataclass, field
import random

import numpy as np
import torch


@dataclass
class TrainingResult:
    episode_rewards: list[float]
    episode_steps: list[int] = field(default_factory=list)
    losses: list = field(default_factory=list)
    eval_returns: list[dict] = field(default_factory=list)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values

    window = max(1, min(int(window), len(values)))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def smooth_result(result, window):
    rewards = np.asarray(result.episode_rewards, dtype=np.float32)
    if len(rewards) == 0:
        return np.array([]), np.array([])

    window = max(1, min(int(window), len(rewards)))
    y = moving_average(rewards, window)

    if result.episode_steps:
        steps = np.asarray(result.episode_steps, dtype=np.float32)
    else:
        steps = np.arange(len(rewards), dtype=np.float32)
    x = steps[window - 1:]
    return x, y


def aggregate_by_steps(results, window, num_points=200):
    curves = [smooth_result(result, window) for result in results]
    curves = [(x, y) for x, y in curves if len(x) > 0 and len(y) > 0]
    if not curves:
        raise ValueError("No non-empty runs to aggregate")

    start = max(float(x[0]) for x, _ in curves)
    end = min(float(x[-1]) for x, _ in curves)
    if end <= start:
        min_len = min(len(y) for _, y in curves)
        grid = np.arange(min_len, dtype=np.float32)
        stacked = np.vstack([y[:min_len] for _, y in curves])
        return grid, stacked.mean(axis=0), stacked.std(axis=0)

    grid = np.linspace(start, end, num_points, dtype=np.float32)
    stacked = np.vstack([np.interp(grid, x, y) for x, y in curves])
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def plot_comparison(results_by_algorithm, window, title, ylabel="Return", num_points=200):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, results in results_by_algorithm.items():
        x, mean, std = aggregate_by_steps(results, window=window, num_points=num_points)
        ax.plot(x, mean, linewidth=2.5, label=name)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def final_window_scores(results_by_algorithm, window):
    scores = {}
    for name, results in results_by_algorithm.items():
        per_seed = []
        for result in results:
            rewards = np.asarray(result.episode_rewards, dtype=np.float32)
            if len(rewards) == 0:
                per_seed.append(np.nan)
            else:
                per_seed.append(float(rewards[-min(window, len(rewards)):].mean()))
        scores[name] = {
            "per_seed": per_seed,
            "mean": float(np.nanmean(per_seed)),
            "std": float(np.nanstd(per_seed)),
        }
    return scores
