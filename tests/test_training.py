"""Leakage, recurrent boundary, and adapter equivalence checks."""

import gymnasium as gym
import copy
import numpy as np
import pytest
import torch

from src.bayes_filter import ExactBayes
from src.baselines import myopic_action, myopic_scores
from src.environment import MarketMakingEnv
from src.interventions import RecurrentActorAdapter, feedforward_probabilities
from src.train import make_model, environment_seeds
from src.wrappers import BeliefWrapper, HistoryWrapper
from src.fast_recurrent import FusedResetRecurrentPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


class PoisonInfo(dict):
    def __getitem__(self, key):
        raise AssertionError("Wrapper must never read info labels")

    def get(self, key, default=None):
        raise AssertionError("Wrapper must never read info labels")


class PublicTraceEnv(gym.Env):
    """Identical public trace with arbitrarily different rewards/diagnostic truth."""

    def __init__(self, private_scale):
        self.observation_space = MarketMakingEnv().observation_space
        self.action_space = gym.spaces.Discrete(10)
        self.private_scale = private_scale

    def reset(self, **kwargs):
        self.t = 0
        return np.array([0, 0, 1, 0, 0, 0, 1, 0, 1], dtype=np.float32), PoisonInfo(V=self.private_scale)

    def step(self, action):
        self.t += 1
        return (np.array([0.3, 0.7, 0, 1, 0, 0, 0, -self.t / 64, 1 - self.t / 64], dtype=np.float32),
                self.private_scale * self.t, self.t == 64, False,
                PoisonInfo(V=self.private_scale, customer_informed=self.private_scale))


@pytest.mark.parametrize("wrapper", [HistoryWrapper, BeliefWrapper])
def test_wrappers_do_not_read_info_or_rewards(wrapper):
    a, b = wrapper(PublicTraceEnv(-10000)), wrapper(PublicTraceEnv(10000))
    np.testing.assert_array_equal(a.reset()[0], b.reset()[0])
    for action in [4, 1, 6, 2]:
        out_a, out_b = a.step(action), b.step(action)
        np.testing.assert_array_equal(out_a[0], out_b[0])
        assert out_a[1] != out_b[1]


def test_history_padding_and_episode_reset():
    env = HistoryWrapper(MarketMakingEnv(), window=8)
    start, _ = env.reset(seed=5)
    assert start.shape == (72,)
    np.testing.assert_array_equal(start[:63], np.zeros(63))
    next_obs, *_ = env.step(4)
    np.testing.assert_array_equal(next_obs[-18:-9], start[-9:])
    reset_obs, _ = env.reset(seed=5)
    np.testing.assert_array_equal(reset_obs, start)


def test_belief_wrapper_conditions_on_submitted_actions_only():
    env = BeliefWrapper(MarketMakingEnv())
    obs, _ = env.reset(seed=31)
    belief = ExactBayes()
    assert obs.shape == (14,)
    np.testing.assert_allclose(obs[:12], belief.posterior)
    for action in [4, 0, 9, 7, 2]:
        obs, _, _, _, info = env.step(action)
        belief.update(action, info["outcome"])
        np.testing.assert_allclose(obs[:12], belief.posterior, rtol=1e-6)
    np.testing.assert_allclose(env.reset(seed=32)[0][:12], np.full(12, 1 / 12))


@pytest.fixture(scope="module")
def recurrent_model():
    config = {"environment": {"horizon": 8}, "training": {
        "n_envs": 2, "n_steps": 8, "batch_size": 8, "n_epochs": 1, "torch_threads": 1}}
    model = make_model(config, "recurrent", seed=13)
    yield model
    model.get_env().close()


def test_architecture_has_separate_critic_and_no_reward_feature(recurrent_model):
    policy = recurrent_model.policy
    assert policy.lstm_actor.input_size == 9
    assert policy.lstm_actor.hidden_size == 64
    assert policy.lstm_actor.num_layers == 1
    assert policy.lstm_critic is not policy.lstm_actor
    assert policy.lstm_critic.hidden_size == 64
    assert not policy.shared_lstm


def test_adapter_matches_official_distribution_and_predict(recurrent_model):
    env = MarketMakingEnv()
    obs, _ = env.reset(seed=3)
    adapter = RecurrentActorAdapter(recurrent_model)
    incoming = None
    for t in range(6):
        result = adapter.process(obs, incoming, episode_start=(t == 0))
        action, expected_state = recurrent_model.predict(
            obs, state=incoming, episode_start=np.array([t == 0]), deterministic=True,
        )
        assert result.choose(deterministic=True) == int(action)
        np.testing.assert_allclose(result.state[0], expected_state[0], atol=1e-7)
        np.testing.assert_allclose(result.state[1], expected_state[1], atol=1e-7)
        initial = incoming if incoming is not None else adapter.initial_state()
        with torch.no_grad():
            dist, _ = recurrent_model.policy.get_distribution(
                torch.tensor(obs[None]), tuple(torch.tensor(x) for x in initial),
                torch.tensor([float(t == 0)]),
            )
        np.testing.assert_allclose(result.probabilities, dist.distribution.probs.numpy()[0], atol=1e-7)
        incoming = result.state
        obs, *_ = env.step(int(action))


def test_episode_start_resets_hidden_and_cell(recurrent_model):
    obs, _ = MarketMakingEnv().reset(seed=10)
    adapter = RecurrentActorAdapter(recurrent_model)
    contaminated = tuple(np.full((1, 1, 64), 7, dtype=np.float32) for _ in range(2))
    reset = adapter.process(obs, state=contaminated, episode_start=True)
    clean = adapter.process(obs, episode_start=True)
    np.testing.assert_allclose(reset.probabilities, clean.probabilities)
    np.testing.assert_allclose(reset.hidden, clean.hidden)
    np.testing.assert_allclose(reset.cell, clean.cell)
    continuing = adapter.process(obs, state=contaminated, episode_start=False)
    assert not np.allclose(continuing.hidden, clean.hidden)


def test_adapter_batch_and_selective_episode_boundaries(recurrent_model):
    obs, _ = MarketMakingEnv().reset(seed=10)
    adapter = RecurrentActorAdapter(recurrent_model)
    incoming = tuple(np.full((1, 2, 64), 3, dtype=np.float32) for _ in range(2))
    batch = adapter.process(np.stack([obs, obs]), state=incoming, episode_start=[True, False])
    assert batch.probabilities.shape == (2, 10)
    assert batch.hidden.shape == batch.cell.shape == (2, 64)
    for i, reset in enumerate([True, False]):
        single = adapter.process(obs, tuple(x[:, i:i + 1] for x in incoming), reset)
        np.testing.assert_allclose(batch.probabilities[i], single.probabilities, atol=1e-7)
        np.testing.assert_allclose(batch.hidden[i], single.hidden, atol=1e-7)


def test_intervention_timing_hidden_and_cell_have_distinct_roles(recurrent_model):
    obs, _ = MarketMakingEnv().reset(seed=9)
    adapter = RecurrentActorAdapter(recurrent_model)
    original = adapter.process(obs)
    h, c = original.state
    cell_patch = adapter.process(obs, after_state=(h, c + 2))
    np.testing.assert_allclose(cell_patch.probabilities, original.probabilities, atol=1e-7)
    next_original = adapter.process(obs, original.state)
    next_patched = adapter.process(obs, cell_patch.state)
    assert not np.allclose(next_original.hidden, next_patched.hidden)
    hidden_patch = adapter.process(obs, after_state=(h + 2, c))
    assert not np.allclose(hidden_patch.probabilities, original.probabilities, atol=1e-7)
    before = adapter.process(obs, before_state=(h + 2, c + 2), episode_start=True)
    clean = adapter.process(obs, episode_start=True)
    np.testing.assert_allclose(before.probabilities, clean.probabilities)


def test_myopic_reference_uses_inventory_cost_and_fee():
    belief = ExactBayes()
    flat = myopic_scores(belief, 0)
    long = myopic_scores(belief, 30, inventory_penalty=1)
    assert not np.allclose(flat, long)
    assert flat[9] == long[9] == 0
    assert myopic_action(belief, 0, fee=2) == 9


@pytest.mark.parametrize("kind,dimension", [("feedforward", 9), ("history", 72), ("belief", 14)])
def test_feedforward_policy_shapes_and_normalized_distribution(kind, dimension):
    config = {"training": {"n_envs": 1, "n_steps": 8, "batch_size": 8}}
    model = make_model(config, kind=kind, seed=11)
    assert model.observation_space.shape == (dimension,)
    obs = model.get_env().reset()
    probabilities = feedforward_probabilities(model, obs)
    assert probabilities.shape == (1, 10)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
    model.get_env().close()


@pytest.mark.parametrize("n_seq,length,layers,reset_mode", [
    (1, 1, 1, "first"), (3, 11, 1, "selective"), (4, 64, 1, "first"),
    (3, 5, 2, "selective"), (2, 16, 1, "none"), (3, 11, 1, "interior"),
])
def test_fused_reset_matches_outputs_and_all_gradients(n_seq, length, layers, reset_mode):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    original_lstm = torch.nn.LSTM(5, 7, num_layers=layers)
    fused_lstm = copy.deepcopy(original_lstm)
    features = torch.randn(n_seq * length, 5, requires_grad=True)
    initial = tuple(torch.randn(layers, n_seq, 7, requires_grad=True) for _ in range(2))
    fused_features = features.detach().clone().requires_grad_()
    fused_initial = tuple(x.detach().clone().requires_grad_() for x in initial)
    starts = torch.zeros((n_seq, length))
    if reset_mode in {"first", "interior"}:
        starts[:, 0] = 1
    if reset_mode == "selective":
        starts[::2, 0] = 1
    if reset_mode == "interior":
        starts[1, length // 2] = 1
    starts = starts.flatten()
    expected, expected_state = RecurrentActorCriticPolicy._process_sequence(
        features, initial, starts, original_lstm,
    )
    actual, actual_state = FusedResetRecurrentPolicy._process_sequence(
        fused_features, fused_initial, starts, fused_lstm,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    for a, b in zip(actual_state, expected_state):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-6)
    weights = torch.randn_like(expected)
    (expected.mul(weights).sum() + sum(x.square().sum() for x in expected_state)).backward()
    (actual.mul(weights).sum() + sum(x.square().sum() for x in actual_state)).backward()
    torch.testing.assert_close(fused_features.grad, features.grad, rtol=2e-5, atol=2e-6)
    for a, b in zip(fused_initial, initial):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)
    for a, b in zip(fused_lstm.parameters(), original_lstm.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)


def test_training_seed_streams_are_disjoint_and_reserved_domains_rejected():
    config = {"evaluation": {"seed": 1000000, "episodes": 1024},
              "analysis": {"seed": 3000000, "episodes": 1024}}
    streams = [environment_seeds(config, seed, 8) for seed in range(11, 16)]
    assert len(set(sum(streams, []))) == 40
    assert streams[0] == list(range(110000, 110008))
    with pytest.raises(ValueError, match="evaluation"):
        environment_seeds(config, 100, 8)
    with pytest.raises(ValueError, match="analysis"):
        environment_seeds(config, 300, 8)
    with pytest.raises(ValueError):
        environment_seeds(config, 11, 10001)


def test_optimizer_seed_preserved_but_environment_seed_remapped(recurrent_model):
    assert recurrent_model.seed == 13
    assert recurrent_model.market_environment_seeds == [130000, 130001]
    assert recurrent_model.get_env()._seeds == [130000, 130001]
