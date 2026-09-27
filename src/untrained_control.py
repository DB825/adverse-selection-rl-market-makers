"""Encode recorded policy observations with an untrained recurrent control.

The original policy determines the trajectories. This module replays precisely
those observations, without choosing actions or consulting diagnostic labels.
It preserves the archive's original row order, regardless of storage ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import time

import numpy as np

from .artifacts import write_json
from .interventions import RecurrentActorAdapter
from .train import load_config, make_model


def encode_observations(episode, times, observations, adapter):
    """Return post-observation hidden/cell states aligned with the input rows.

    Each episode must include exactly one observation at times 0 through its
    final recorded time. Episodes may differ in length. Nothing is inferred
    from an outcome, action, reward, target, or latent-state diagnostic column.
    """
    episode = np.asarray(episode)
    times = np.asarray(times)
    observations = np.asarray(observations, dtype=np.float32)
    if (episode.ndim != 1 or times.shape != episode.shape
            or observations.ndim != 2 or len(observations) != len(episode)
            or not len(episode)):
        raise ValueError("Expected nonempty episode/t vectors and one observation per row")
    if (not np.issubdtype(episode.dtype, np.integer)
            or not np.issubdtype(times.dtype, np.integer)
            or np.any(times < 0) or not np.isfinite(observations).all()):
        raise ValueError("Episode IDs and nonnegative times must be integers; observations must be finite")
    unique_episodes = np.unique(episode)
    for eid in unique_episodes:
        trajectory_times = np.sort(times[episode == eid])
        if not np.array_equal(trajectory_times, np.arange(len(trajectory_times))):
            raise ValueError(f"Episode {eid} must have exactly one row at each time beginning at zero")
    global_state = adapter.initial_state(len(unique_episodes))
    width = global_state[0].shape[-1]
    hidden = np.empty((len(episode), width), dtype=np.float32)
    cell = np.empty_like(hidden)
    for t in np.unique(times):
        rows = np.flatnonzero(times == t)
        rows = rows[np.argsort(episode[rows])]
        slots = np.searchsorted(unique_episodes, episode[rows])
        incoming = tuple(part[:, slots, :] for part in global_state)
        step = adapter.process(observations[rows], state=incoming,
                               episode_start=np.full(len(rows), t == 0))
        hidden[rows], cell[rows] = step.hidden, step.cell
        for all_episodes_state, updated in zip(global_state, step.state):
            all_episodes_state[:, slots, :] = updated
    return hidden, cell


def collect_untrained_states(dataset, config="configs/pilot.yaml", seed=11, output=None):
    """Construct, but never train, a recurrent policy and replay public inputs."""
    source = Path(dataset)
    output = Path(output) if output is not None else source.parent / "untrained_states.npz"
    if output.suffix.lower() != ".npz":
        output = output / "untrained_states.npz"
    if source.resolve() == output.resolve():
        raise ValueError("Control output must differ from the source activation archive")
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing control result: {output}")
    started = time.perf_counter()
    resolved_config = load_config(config)
    # Select these three named arrays explicitly. Other arrays, including
    # hidden simulator labels and trained activations, are never loaded.
    with np.load(source, allow_pickle=False) as data:
        episode = data["episode"].copy()
        times = data["t"].copy()
        observations = data["obs"].copy()
    model = make_model(resolved_config, kind="recurrent", seed=seed)
    try:
        adapter = RecurrentActorAdapter(model)
        hidden, cell = encode_observations(episode, times, observations, adapter)
        if model.num_timesteps != 0:
            raise AssertionError("An untrained control must have zero training timesteps")
    finally:
        model.get_env().close()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, episode=episode, t=times,
                        untrained_hidden=hidden, untrained_cell=cell)
    metadata = dict(
        dataset=str(source.resolve()), output=str(output.resolve()),
        dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        config=resolved_config, initialization_seed=int(seed), training_timesteps=0,
        rows=len(episode), episodes=len(np.unique(episode)),
        minimum_time=int(times.min()), maximum_time=int(times.max()),
        observation_inputs=["obs"], indexing_inputs=["episode", "t"],
        source_row_order_preserved=True,
        timing="after processing each current observation, before actor MLP/action head",
        trajectory_source="Recorded trained-policy observations; control never selects actions",
        shapes={"untrained_hidden": list(hidden.shape), "untrained_cell": list(cell.shape)},
        dependencies={name: importlib.metadata.version(name)
                      for name in ("numpy", "torch", "stable-baselines3", "sb3-contrib")},
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json(metadata_path, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output")
    result = collect_untrained_states(**vars(parser.parse_args()))
    print({key: result[key] for key in ("output", "rows", "episodes", "initialization_seed", "elapsed_seconds")})


if __name__ == "__main__":
    main()
