import functools
import os
from datetime import datetime

import brax
import flax
import jax
import matplotlib.pyplot as plt
from brax import envs
from brax.io import html, json, model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from jax import numpy as jnp
from rich.pretty import pprint
from tqdm import tqdm


def mk_env(env_name, backend):

    env_name = "humanoid"  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    backend = "positional"  # @param ['generalized', 'positional', 'spring']

    env = envs.get_environment(env_name=env_name, backend=backend)
    state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

    print("created env")
    return env, state


# Training
"""
Brax provides out of the box the following training algorithms:

* [Proximal policy optimization]
* [Soft actor-critic]
* [Evolutionary strategy]
* [Analytic policy gradients]
* [Augmented random search]

Trainers take as input an environment function and some hyperparameters, and return an inference function to operate the environment.

"""


def train(env, env_name, steps, nevals=10):

    # We determined some reasonable hyperparameters offline and share them here.
    train_fn = {
        "ant": functools.partial(
            ppo.train,
            num_timesteps=steps,
            num_evals=nevals,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=5,
            num_minibatches=32,
            num_updates_per_batch=4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=4096,
            batch_size=2048,
            seed=1,
        ),
        "humanoid": functools.partial(
            ppo.train,
            num_timesteps=steps,
            num_evals=nevals,
            reward_scaling=0.1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=10,
            num_minibatches=32,
            num_updates_per_batch=8,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            num_envs=2048,
            batch_size=1024,
            seed=1,
        ),
        "reacher": functools.partial(
            ppo.train,
            num_timesteps=steps,
            num_evals=nevals,
            reward_scaling=5,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=4,
            unroll_length=50,
            num_minibatches=32,
            num_updates_per_batch=8,
            discounting=0.95,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            num_envs=2048,
            batch_size=256,
            max_devices_per_host=8,
            seed=1,
        ),
        "humanoidstandup": functools.partial(
            ppo.train,
            num_timesteps=steps,
            num_evals=nevals,
            reward_scaling=0.1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=15,
            num_minibatches=32,
            num_updates_per_batch=8,
            discounting=0.97,
            learning_rate=6e-4,
            entropy_cost=1e-2,
            num_envs=2048,
            batch_size=1024,
            seed=1,
        ),
    }[env_name]

    max_y = {
        "ant": 8000,
        "halfcheetah": 8000,
        "hopper": 2500,
        "humanoid": 13000,
        "humanoidstandup": 75_000,
        "reacher": 5,
        "walker2d": 5000,
        "pusher": 0,
    }[env_name]
    min_y = {"reacher": -100, "pusher": -150}.get(env_name, 0)

    xdata, ydata = [], []
    times = [datetime.now()]

    bar = tqdm(range(steps))
    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"])
        pprint(num_steps)
        pprint(metrics)
        bar.update(num_steps - bar.n)
        plt.xlim([0, train_fn.keywords["num_timesteps"]])
        plt.ylim([min_y, max_y])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.plot(xdata, ydata)
        plt.show()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    """
    The trainers return an inference function, parameters, 
    and the final set of metrics gathered during evaluation.

    # Saving and Loading Policies
    Brax can save and load trained policies:
    """

    model.save_params("/tmp/params", params)
    params = model.load_params("/tmp/params")
    inference_fn = make_inference_fn(params)

    """The trainers return an inference function, parameters, and the final set of metrics gathered during evaluation.
    # Saving and Loading Policies
    Brax can save and load trained policies:
    """

    return inference_fn


def val(env_name, backend, inference_fn):

    # @title Visualizing a trajectory of the learned inference function

    # create an env with auto-reset
    env = envs.create(env_name=env_name, backend=backend)

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    infer = jax.jit(inference_fn)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = reset(rng=rng)
    for _ in tqdm(range(1000)):
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = infer(state.obs, act_rng)
        state = step(state, act)

    # pprint(rollout[-1]) # State()
    h = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout)
    print()
    with open("output.html", "w") as f:
        f.write(h)


def main():

    steps = 100_000_000

    env, state = mk_env("humanoid", "positional")
    inference_fn = train(env, "humanoid", steps, nevals=10)
    val("humanoid", "positional", inference_fn)


if __name__ == "__main__":
    main()
