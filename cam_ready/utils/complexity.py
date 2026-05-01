import math


def mlp_macs(layer_sizes):
    return int(sum(in_dim * out_dim for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])))


def mlp_params(layer_sizes):
    return int(sum(in_dim * out_dim + out_dim for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])))


def bytes_to_mb(num_bytes):
    return num_bytes / (1024 ** 2)


def offpolicy_update_count(total_steps, update_after, update_every):
    if total_steps <= update_after:
        return 0
    first = update_after
    if first % update_every != 0:
        first += update_every - (first % update_every)
    if first >= total_steps:
        return 0
    update_events = ((total_steps - 1 - first) // update_every) + 1
    return int(update_events * update_every)


def dqn_complexity(obs_dim=4,
                   act_dim=2,
                   hidden_sizes=(128, 128),
                   total_steps=50_000,
                   batch_size=256,
                   step_skipping=2,
                   start_step=100,
                   memory_size=100_000):
    q_layers = [obs_dim, *hidden_sizes, act_dim]
    q_macs = mlp_macs(q_layers)
    q_params = mlp_params(q_layers)
    updates = max(0, math.floor((total_steps - start_step) / step_skipping))
    per_update_macs = batch_size * 4 * q_macs
    total_macs = total_steps * q_macs + updates * per_update_macs
    replay_bytes = memory_size * ((2 * obs_dim + 3) * 4)
    param_bytes = 2 * q_params * 4

    return {
        "algorithm": "DQN",
        "big_o_time": "O(E*Cq + U*B*Cq)",
        "big_o_memory": "O(R*(S+A) + |Q|)",
        "env_steps": total_steps,
        "gradient_updates": updates,
        "parameters": 2 * q_params,
        "per_update_mac_proxy": per_update_macs,
        "total_mac_proxy": int(total_macs),
        "storage_mb": round(bytes_to_mb(replay_bytes + param_bytes), 3),
    }


def ppo_complexity(obs_dim,
                   act_dim,
                   hidden_sizes=(64, 64),
                   total_steps=50_000,
                   n_actors=4,
                   T=128,
                   K=4,
                   batch_size=64,
                   continuous=False):
    actor_layers = [obs_dim, *hidden_sizes, act_dim]
    critic_layers = [obs_dim, *hidden_sizes, 1]
    actor_macs = mlp_macs(actor_layers)
    critic_macs = mlp_macs(critic_layers)
    actor_params = mlp_params(actor_layers)
    critic_params = mlp_params(critic_layers)
    if continuous:
        actor_params += act_dim

    steps_per_iteration = T * n_actors
    iterations = max(1, total_steps // steps_per_iteration)
    effective_steps = iterations * steps_per_iteration
    samples_per_iteration = steps_per_iteration
    updates = iterations * K * math.ceil(samples_per_iteration / batch_size)

    rollout_macs = effective_steps * (actor_macs + critic_macs)
    update_macs = K * effective_steps * 3 * (actor_macs + critic_macs)
    buffer_values = obs_dim + act_dim + 6
    rollout_bytes = T * n_actors * buffer_values * 4
    param_bytes = (actor_params + critic_params) * 4

    return {
        "algorithm": "PPO",
        "big_o_time": "O(E*(Cpi+Cv) + K*E*(Cpi+Cv))",
        "big_o_memory": "O(T*N*(S+A) + |pi| + |V|)",
        "env_steps": effective_steps,
        "gradient_updates": updates,
        "parameters": actor_params + critic_params,
        "per_update_mac_proxy": batch_size * 3 * (actor_macs + critic_macs),
        "total_mac_proxy": int(rollout_macs + update_macs),
        "storage_mb": round(bytes_to_mb(rollout_bytes + param_bytes), 3),
    }


def sac_actor_macs(obs_dim, act_dim, hidden_sizes):
    trunk_layers = [obs_dim, *hidden_sizes]
    return mlp_macs(trunk_layers) + 2 * hidden_sizes[-1] * act_dim


def sac_actor_params(obs_dim, act_dim, hidden_sizes):
    trunk_layers = [obs_dim, *hidden_sizes]
    return mlp_params(trunk_layers) + 2 * (hidden_sizes[-1] * act_dim + act_dim)


def sac_complexity(obs_dim=3,
                   act_dim=1,
                   hidden_sizes=(256, 256),
                   total_steps=25_000,
                   batch_size=100,
                   update_after=1_000,
                   update_every=50,
                   replay_size=100_000):
    actor_macs = sac_actor_macs(obs_dim, act_dim, hidden_sizes)
    q_layers = [obs_dim + act_dim, *hidden_sizes, 1]
    q_macs = mlp_macs(q_layers)
    actor_params = sac_actor_params(obs_dim, act_dim, hidden_sizes)
    q_params = mlp_params(q_layers)
    updates = offpolicy_update_count(total_steps, update_after, update_every)

    per_update_macs = batch_size * (4 * actor_macs + 14 * q_macs)
    action_macs = total_steps * actor_macs
    total_macs = action_macs + updates * per_update_macs
    replay_bytes = replay_size * ((2 * obs_dim + act_dim + 3) * 4)
    param_bytes = 2 * (actor_params + 2 * q_params) * 4

    return {
        "algorithm": "SAC",
        "big_o_time": "O(E*Cpi + U*B*(Cpi+Cq))",
        "big_o_memory": "O(R*(S+A) + |pi| + 2|Q|)",
        "env_steps": total_steps,
        "gradient_updates": updates,
        "parameters": 2 * (actor_params + 2 * q_params),
        "per_update_mac_proxy": per_update_macs,
        "total_mac_proxy": int(total_macs),
        "storage_mb": round(bytes_to_mb(replay_bytes + param_bytes), 3),
    }


def td3_complexity(obs_dim=3,
                   act_dim=1,
                   hidden_sizes=(256, 256),
                   total_steps=25_000,
                   batch_size=100,
                   update_after=1_000,
                   update_every=50,
                   policy_delay=2,
                   replay_size=100_000):
    actor_layers = [obs_dim, *hidden_sizes, act_dim]
    q_layers = [obs_dim + act_dim, *hidden_sizes, 1]
    actor_macs = mlp_macs(actor_layers)
    q_macs = mlp_macs(q_layers)
    actor_params = mlp_params(actor_layers)
    q_params = mlp_params(q_layers)
    updates = offpolicy_update_count(total_steps, update_after, update_every)

    critic_update_macs = batch_size * (actor_macs + 8 * q_macs)
    actor_update_macs = batch_size * (3 * actor_macs + 3 * q_macs)
    actor_updates = updates // policy_delay
    total_macs = total_steps * actor_macs + updates * critic_update_macs + actor_updates * actor_update_macs
    replay_bytes = replay_size * ((2 * obs_dim + act_dim + 3) * 4)
    param_bytes = 2 * (actor_params + 2 * q_params) * 4

    return {
        "algorithm": "TD3",
        "big_o_time": "O(E*Cpi + U*B*Cq + (U/d)*B*(Cpi+Cq))",
        "big_o_memory": "O(R*(S+A) + |pi| + 2|Q|)",
        "env_steps": total_steps,
        "gradient_updates": updates,
        "parameters": 2 * (actor_params + 2 * q_params),
        "per_update_mac_proxy": critic_update_macs + actor_update_macs / policy_delay,
        "total_mac_proxy": int(total_macs),
        "storage_mb": round(bytes_to_mb(replay_bytes + param_bytes), 3),
    }


def discrete_cartpole_complexity(total_env_steps=50_000):
    return [
        dqn_complexity(total_steps=total_env_steps),
        ppo_complexity(
            obs_dim=4,
            act_dim=2,
            hidden_sizes=(64, 64),
            total_steps=total_env_steps,
            n_actors=4,
            T=128,
            K=4,
            batch_size=64,
            continuous=False,
        ),
    ]


def continuous_pendulum_complexity(total_env_steps=25_000):
    return [
        ppo_complexity(
            obs_dim=3,
            act_dim=1,
            hidden_sizes=(64, 64),
            total_steps=total_env_steps,
            n_actors=4,
            T=256,
            K=5,
            batch_size=64,
            continuous=True,
        ),
        sac_complexity(total_steps=total_env_steps),
        td3_complexity(total_steps=total_env_steps),
    ]
