"""Reproducible PPO and RecurrentPPO training with raw economic rewards."""

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .environment import MarketMakingEnv
from .artifacts import source_hashes
from .fast_recurrent import FusedResetRecurrentPolicy
from .wrappers import wrap_observations


POLICIES = ("recurrent", "feedforward", "history", "belief")


def load_config(config):
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    return copy.deepcopy(config)


def make_env(config, kind="recurrent"):
    config = load_config(config)
    env_config = config.get("environment", {})
    env = MarketMakingEnv(**env_config)
    return wrap_observations(
        env, kind, history_window=config.get("training", {}).get("history_window", 8),
        regimes=env_config.get("regimes"), prior=env_config.get("prior"),
    )


def environment_seeds(config, seed, n_envs):
    """Keep optimizer seeds small while assigning disjoint simulator streams."""
    if not 1 <= n_envs <= 10000 or int(seed) != seed or seed < 0:
        raise ValueError("Require nonnegative integer seed and 1..10000 environments")
    first = int(seed) * 10000
    seeds = list(range(first, first + n_envs))
    for domain in ("evaluation", "analysis"):
        settings = config.get(domain, {})
        reserved = settings.get("seed")
        if reserved is not None:
            end = int(reserved) + int(settings.get("episodes", 1))
            if first < end and first + n_envs > int(reserved):
                raise ValueError(f"Training environment seed range overlaps {domain} seed range")
    return seeds


def make_model(config, kind="recurrent", seed=11, env=None):
    """Construct an untrained model; useful for smoke tests and random controls."""
    config = load_config(config)
    train_config = config.get("training", {})
    torch.set_num_threads(int(train_config.get("torch_threads", 1)))
    torch.backends.mkldnn.enabled = bool(train_config.get("torch_mkldnn", True))
    if kind not in POLICIES:
        raise ValueError(f"kind must be one of {POLICIES}")
    if env is None:
        env = DummyVecEnv([
            lambda: Monitor(make_env(config, kind))
            for _ in range(int(train_config.get("n_envs", 8)))
        ])
    kwargs = dict(
        learning_rate=float(train_config.get("learning_rate", 3e-4)),
        n_steps=int(train_config.get("n_steps", 128)),
        batch_size=int(train_config.get("batch_size", 64)),
        n_epochs=int(train_config.get("n_epochs", 10)),
        gamma=float(train_config.get("gamma", 1.0)),
        gae_lambda=float(train_config.get("gae_lambda", 0.95)),
        clip_range=float(train_config.get("clip_range", 0.2)),
        ent_coef=float(train_config.get("ent_coef", 0.0)),
        vf_coef=float(train_config.get("vf_coef", 0.5)),
        max_grad_norm=float(train_config.get("max_grad_norm", 0.5)),
        seed=int(seed), device=train_config.get("device", "cpu"), verbose=0,
    )
    if kwargs["gamma"] != 1.0:
        raise ValueError("This pilot requires gamma=1 to match its accounting objective")
    if kind == "recurrent":
        model = RecurrentPPO(
            FusedResetRecurrentPolicy if train_config.get("fused_lstm_reset", False) else "MlpLstmPolicy", env,
            policy_kwargs=dict(lstm_hidden_size=64, n_lstm_layers=1,
                               shared_lstm=False, enable_critic_lstm=True,
                               net_arch=dict(pi=[64], vf=[64])), **kwargs,
        )
    else:
        model = PPO("MlpPolicy", env,
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])), **kwargs)
    # SB3's algorithm setup first calls vec_env.seed(seed), producing streams
    # seed..seed+n_envs-1. Consecutive optimizer seeds would share streams.
    # Override only environment seeds AFTER setup, BEFORE learn/reset.
    streams = environment_seeds(config, int(seed), model.get_env().num_envs)
    model.get_env().seed(streams[0])
    model.market_environment_seeds = streams
    return model


class EconomicDiagnostics(BaseCallback):
    """Log completed-episode economics; simulator diagnostics stay out of obs."""

    def __init__(self):
        super().__init__()
        self.episodes = []

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.episodes.append({
                    name: float(info[name]) for name in
                    ("objective", "profit", "inventory_penalty_total") if name in info
                })
        return True

    def _on_rollout_end(self):
        if self.episodes:
            for name in ("objective", "profit", "inventory_penalty_total"):
                values = [row[name] for row in self.episodes if name in row]
                if values:
                    self.logger.record(f"economics/{name}_mean", float(np.mean(values)))
            self.logger.record("economics/episodes", len(self.episodes))
            self.episodes.clear()


def _dependency_versions():
    packages = ("numpy", "gymnasium", "torch", "stable-baselines3", "sb3-contrib",
                "scikit-learn", "matplotlib", "tensorboard", "pytest", "PyYAML", "pandas")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train(config, kind="recurrent", seed=11, output=None, steps=None):
    """Train, persist checkpoint/config/metrics, and return (model, metadata)."""
    config = load_config(config)
    output = Path(output or f"results/{kind}/seed_{seed}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.zip").exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output / 'model.zip'}")
    requested_steps = int(steps if steps is not None else config["training"]["total_timesteps"])
    if requested_steps < 1:
        raise ValueError("total_timesteps must be positive")
    model = make_model(config, kind=kind, seed=seed)
    logger = configure(str(output), ["csv", "tensorboard"])
    model.set_logger(logger)
    # SB3 clips immediately before optimizer.step. This hook measures the norm
    # after clipping; it cannot diagnose the distribution of pre-clip norms.
    def observe_gradients(optimizer, args, kwargs):
        grad_squares = [p.grad.detach().square().sum() for p in model.policy.parameters()
                        if p.grad is not None]
        if grad_squares:
            norm = torch.stack(grad_squares).sum().sqrt().item()
            model.logger.record_mean("train/post_clip_gradient_norm", norm)

    hook = model.policy.optimizer.register_step_pre_hook(observe_gradients)
    parameter_count = sum(p.numel() for p in model.policy.parameters())
    metadata = {
        "policy": kind, "seed": int(seed), "requested_steps": requested_steps,
        "environment_seeds": model.market_environment_seeds,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dependencies": _dependency_versions(), "python": platform.python_version(),
        "git_revision": _git_revision(), "device": str(model.device),
        "source_hashes": source_hashes(),
        "torch_threads": torch.get_num_threads(), "parameter_count": parameter_count,
        "torch_mkldnn": torch.backends.mkldnn.enabled,
        "fused_lstm_reset": isinstance(model.policy, FusedResetRecurrentPolicy),
        "checkpoint": str((output / "model.zip").resolve()),
        "reward_normalization": False, "reward_as_policy_input": False,
        "gradient_diagnostic": "L2 norm across actor+critic gradients AFTER global clipping, before optimizer step",
        "architecture": {
            "actor": "LSTM(9,64), one layer; tanh MLP[64]; categorical10" if kind == "recurrent"
                     else "tanh MLP[64,64]; categorical10",
            "critic": "separate LSTM(9,64), one layer; tanh MLP[64]; scalar value" if kind == "recurrent"
                      else "separate tanh MLP[64,64]; scalar value",
            "input_dimension": int(np.prod(model.observation_space.shape)),
        },
        "notes": ["gamma=1 matches finite-horizon undiscounted reward accounting",
                  "8 serial DummyVecEnv instances; 128 steps each gives 1024 transitions/update by default",
                  "PPO rollout length reduced from SB3 PPO default 2048 per environment; RecurrentPPO default is 128",
                  f"batch_size={model.batch_size}; PPO default64 and RecurrentPPO default128; pilot retains64 after equivalent fused-reset optimization",
                  "Default PPO GAE lambda .95 introduces normal critic-based advantage estimation; raw rewards are unchanged",
                  "No observation/reward normalization wrappers; no rewards or diagnostic labels are policy inputs"],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=requested_steps, callback=EconomicDiagnostics(), log_interval=1)
        model.logger.dump(step=model.num_timesteps)  # Include last optimization diagnostics.
        model.save(output / "model")
    finally:
        hook.remove()
    metadata.update(actual_steps=int(model.num_timesteps), elapsed_seconds=time.perf_counter() - start)
    metadata["steps_per_second"] = metadata["actual_steps"] / metadata["elapsed_seconds"]
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model.get_env().close()
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--policy", choices=POLICIES, default="recurrent")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    _, metadata = train(args.config, args.policy, args.seed, args.output, args.steps)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
