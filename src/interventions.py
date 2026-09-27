"""Small evaluation adapter for actor-state logging and future interventions.

No training algorithm is replaced. The adapter follows sb3-contrib's actor
feature extractor -> actor LSTM -> actor MLP -> categorical head exactly.
Both h and c are logged *after processing the current observation*. A patched
post-observation h changes the immediate head input and future memory; a
patched c alone changes future memory but not the immediate action head.

State interventions are interfaces only: they do not establish causal use,
matching quality, or whether a patched state lies on the policy's manifold.
"""

from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3.common.policies import BaseModel


@dataclass
class ActorStep:
    probabilities: np.ndarray
    hidden: np.ndarray
    cell: np.ndarray
    state: tuple[np.ndarray, np.ndarray]

    def choose(self, rng=None, deterministic=False):
        if self.probabilities.ndim != 1:
            raise ValueError("choose() requires one observation")
        if deterministic:
            return int(self.probabilities.argmax())
        if rng is None:
            raise ValueError("Provide an explicit evaluation RNG for stochastic actions")
        probabilities = self.probabilities.astype(float)
        return int(rng.choice(len(probabilities), p=probabilities / probabilities.sum()))


class RecurrentActorAdapter:
    """Stateless adapter; caller carries ``ActorStep.state`` to the next step.

    ``before_state`` replaces incoming h,c before episode reset/observation;
    an episode-start reset therefore intentionally overrides such a patch.
    ``after_state`` replaces outgoing h,c after the observation, before the
    actor MLP and categorical action head. Arrays have (layers,batch,units).
    Only the default non-projected LSTM architecture is supported.
    """

    def __init__(self, model):
        self.model = model
        self.policy = model.policy
        if not hasattr(self.policy, "lstm_actor"):
            raise TypeError("RecurrentActorAdapter requires a recurrent actor")
        if self.policy.lstm_actor.proj_size or self.policy.lstm_actor.bidirectional:
            raise ValueError("Projected or bidirectional LSTMs are not supported")
        self.policy.set_training_mode(False)

    def initial_state(self, batch_size=1):
        shape = (self.policy.lstm_actor.num_layers, batch_size,
                 self.policy.lstm_actor.hidden_size)
        return (np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32))

    @torch.no_grad()
    def process(self, obs, state=None, episode_start=False, before_state=None, after_state=None):
        policy = self.policy
        obs_tensor, vectorized = policy.obs_to_tensor(np.asarray(obs, dtype=np.float32))
        batch_size = obs_tensor.shape[0]
        incoming = before_state if before_state is not None else state
        if incoming is None:
            incoming = self.initial_state(batch_size)
        expected_shape = self.initial_state(batch_size)[0].shape

        def as_state(pair):
            if len(pair) != 2 or any(np.asarray(x).shape != expected_shape for x in pair):
                raise ValueError(f"Expected h,c each of shape {expected_shape}")
            return tuple(torch.as_tensor(x, dtype=torch.float32, device=policy.device) for x in pair)

        incoming_tensor = as_state(incoming)
        starts = np.broadcast_to(np.asarray(episode_start, dtype=np.float32), (batch_size,)).copy()
        starts_tensor = torch.as_tensor(starts, device=policy.device)
        # BaseModel performs SB3 preprocessing, then the same actor extractor.
        features = BaseModel.extract_features(policy, obs_tensor, policy.pi_features_extractor)
        recurrent_output, outgoing = policy._process_sequence(
            features, incoming_tensor, starts_tensor, policy.lstm_actor,
        )
        if after_state is not None:
            outgoing = as_state(after_state)
            recurrent_output = outgoing[0][-1]
        latent = policy.mlp_extractor.forward_actor(recurrent_output)
        dist = policy._get_action_dist_from_latent(latent)
        probabilities = dist.distribution.probs.cpu().numpy().copy()
        outgoing_np = tuple(x.cpu().numpy().copy() for x in outgoing)
        hidden = outgoing_np[0][-1].copy()
        cell = outgoing_np[1][-1].copy()
        if not vectorized:
            probabilities, hidden, cell = probabilities[0], hidden[0], cell[0]
        return ActorStep(probabilities, hidden, cell, outgoing_np)


@torch.no_grad()
def feedforward_probabilities(model, observation):
    model.policy.set_training_mode(False)
    obs_tensor, vectorized = model.policy.obs_to_tensor(observation)
    probabilities = model.policy.get_distribution(obs_tensor).distribution.probs.cpu().numpy()
    return probabilities if vectorized else probabilities[0]
