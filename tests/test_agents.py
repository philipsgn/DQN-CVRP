"""Unit tests for src/agents/: the masked bootstrap-target computation,
Double DQN's target/online split, and the replay buffer — none of which had
dedicated coverage before this file (found in the 2026-08-05 audit; the
masking logic is called "the single most consequential design decision" in
Chapter 3 §3.3.5, but was previously exercised only implicitly, via full
training runs converging)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.dqn import DQNAgent, DQNConfig
from src.agents.networks import masked_q
from src.agents.replay import ReplayBuffer


# ------------------------------------------------------------- masked_q


def test_masked_q_sets_infeasible_to_negative_infinity():
    q = torch.tensor([[1.0, 5.0, -2.0]])
    mask = torch.tensor([[True, False, True]])
    out = masked_q(q, mask)
    assert out[0, 1].item() == float("-inf")
    assert out[0, 0].item() == 1.0
    assert out[0, 2].item() == -2.0


def test_masked_q_argmax_never_picks_infeasible_action_even_if_highest():
    # The masked-out action has the largest raw value in both rows; a
    # correct masked argmax must never select it.
    q = torch.tensor([[1.0, 100.0, -2.0], [0.0, 3.0, 9.0]])
    mask = torch.tensor([[True, False, True], [True, True, False]])
    out = masked_q(q, mask)
    picks = out.argmax(dim=1)
    assert picks[0].item() == 0  # not 1, despite q=100 there
    assert picks[1].item() == 1  # not 2, despite q=9 there


# ------------------------------------------------------------- DQNAgent target masking


def test_train_step_target_excludes_masked_next_action():
    """Regression test for the design decision Chapter 3 calls out as most
    consequential: if the mask were dropped from the bootstrap target (the
    "classic bug that stops masked DQN from converging", per dqn.py's own
    module docstring), a masked-out next action with an artificially large
    raw Q-value would leak into the target. This constructs exactly that
    trap — mask out whichever next-action the (frozen, pre-step) target
    network currently rates highest — and asserts the loss actually
    computed by `_train_step` matches an independent, direct recomputation
    of the masked target, not the unmasked one."""
    obs_dim, n_actions, batch = 4, 3, 16
    config = DQNConfig(batch_size=batch, warmup_steps=0, double_dqn=False,
                        gamma=0.9)
    agent = DQNAgent(obs_dim, n_actions, config, seed=0)

    next_obs = np.random.default_rng(1).uniform(size=(batch, obs_dim)).astype(np.float32)
    obs = np.random.default_rng(2).uniform(size=(batch, obs_dim)).astype(np.float32)
    action = np.zeros(batch, dtype=np.int64)
    reward = np.full(batch, 0.5, dtype=np.float32)
    done = np.zeros(batch, dtype=np.float32)

    with torch.no_grad():
        raw_next_q = agent.target(torch.as_tensor(next_obs))
    best_per_row = raw_next_q.argmax(dim=1).numpy()

    # Mask out each row's own highest-scoring next action, so a correct
    # masked target must never equal the raw (unmasked) max for that row.
    next_mask = np.ones((batch, n_actions), dtype=bool)
    next_mask[np.arange(batch), best_per_row] = False

    fixed_batch = dict(obs=obs, action=action, reward=reward,
                        next_obs=next_obs, next_mask=next_mask, done=done)
    agent.buffer.sample = lambda n: fixed_batch  # deterministic batch, no need to fill buffer for real

    with torch.no_grad():
        q_sa_before = agent.q(torch.as_tensor(obs)).gather(
            1, torch.as_tensor(action).unsqueeze(1)).squeeze(1)
        masked_next_q = masked_q(raw_next_q, torch.as_tensor(next_mask))
        expected_target = torch.as_tensor(reward) + config.gamma * masked_next_q.max(dim=1).values
        expected_loss = torch.nn.SmoothL1Loss()(q_sa_before, expected_target).item()

        # The bug this test guards against: if masking were skipped, the
        # target would use the raw (unmasked) max instead.
        unmasked_target = torch.as_tensor(reward) + config.gamma * raw_next_q.max(dim=1).values
        buggy_loss = torch.nn.SmoothL1Loss()(q_sa_before, unmasked_target).item()

    actual_loss = agent._train_step()

    assert actual_loss == pytest.approx(expected_loss, abs=1e-4)
    # The two losses must differ (by construction, the masked-out action was
    # the highest-scoring one for every row) — otherwise this test wouldn't
    # actually be exercising the masking logic at all.
    assert abs(expected_loss - buggy_loss) > 1e-6


def test_double_dqn_uses_online_network_to_pick_masked_argmax():
    """Double DQN's defining property: the *online* network selects which
    next-action is best (respecting the mask), and the *target* network only
    supplies that action's value — never the target network's own argmax."""
    obs_dim, n_actions, batch = 4, 4, 8
    config = DQNConfig(batch_size=batch, warmup_steps=0, double_dqn=True, gamma=0.9)
    agent = DQNAgent(obs_dim, n_actions, config, seed=3)
    # Force target and online networks to disagree by nudging online weights.
    with torch.no_grad():
        for p in agent.q.parameters():
            p.add_(0.1)

    next_obs = np.random.default_rng(4).uniform(size=(batch, obs_dim)).astype(np.float32)
    obs = np.random.default_rng(5).uniform(size=(batch, obs_dim)).astype(np.float32)
    action = np.zeros(batch, dtype=np.int64)
    reward = np.zeros(batch, dtype=np.float32)
    done = np.zeros(batch, dtype=np.float32)
    next_mask = np.ones((batch, n_actions), dtype=bool)

    fixed_batch = dict(obs=obs, action=action, reward=reward,
                        next_obs=next_obs, next_mask=next_mask, done=done)
    agent.buffer.sample = lambda n: fixed_batch

    with torch.no_grad():
        online_next = masked_q(agent.q(torch.as_tensor(next_obs)), torch.as_tensor(next_mask))
        best = online_next.argmax(dim=1, keepdim=True)
        target_next = masked_q(agent.target(torch.as_tensor(next_obs)), torch.as_tensor(next_mask))
        expected_next_q = target_next.gather(1, best).squeeze(1)
        # The wrong-but-plausible alternative: target network picks its own argmax.
        wrong_next_q = target_next.max(dim=1).values
        q_sa_before = agent.q(torch.as_tensor(obs)).gather(
            1, torch.as_tensor(action).unsqueeze(1)).squeeze(1)
        expected_loss = torch.nn.SmoothL1Loss()(
            q_sa_before, torch.as_tensor(reward) + config.gamma * expected_next_q).item()
        wrong_loss = torch.nn.SmoothL1Loss()(
            q_sa_before, torch.as_tensor(reward) + config.gamma * wrong_next_q).item()

    actual_loss = agent._train_step()
    assert actual_loss == pytest.approx(expected_loss, abs=1e-4)
    if abs(expected_next_q - wrong_next_q).max().item() > 1e-6:
        assert abs(expected_loss - wrong_loss) > 1e-9


def test_done_transitions_zero_out_bootstrap():
    """A terminal transition's target must be just the reward, regardless of
    next_mask/next_obs content (which are meaningless past termination)."""
    obs_dim, n_actions, batch = 3, 2, 8
    config = DQNConfig(batch_size=batch, warmup_steps=0, gamma=0.99)
    agent = DQNAgent(obs_dim, n_actions, config, seed=6)
    obs = np.zeros((batch, obs_dim), dtype=np.float32)
    next_obs = np.random.default_rng(7).uniform(size=(batch, obs_dim)).astype(np.float32)
    action = np.zeros(batch, dtype=np.int64)
    reward = np.full(batch, 2.0, dtype=np.float32)
    done = np.ones(batch, dtype=np.float32)
    next_mask = np.ones((batch, n_actions), dtype=bool)
    fixed_batch = dict(obs=obs, action=action, reward=reward,
                        next_obs=next_obs, next_mask=next_mask, done=done)
    agent.buffer.sample = lambda n: fixed_batch
    with torch.no_grad():
        q_sa_before = agent.q(torch.as_tensor(obs)).gather(
            1, torch.as_tensor(action).unsqueeze(1)).squeeze(1)
        expected_loss = torch.nn.SmoothL1Loss()(
            q_sa_before, torch.as_tensor(reward)).item()
    actual_loss = agent._train_step()
    assert actual_loss == pytest.approx(expected_loss, abs=1e-4)


# ------------------------------------------------------------- ReplayBuffer


def test_replay_buffer_overwrites_oldest_at_capacity():
    buf = ReplayBuffer(capacity=4, obs_dim=1, n_actions=2, seed=0)
    for i in range(6):
        buf.push(obs=[float(i)], action=0, reward=float(i),
                  next_obs=[float(i)], next_mask=[True, False], done=False)
    assert len(buf) == 4
    # Slots wrap circularly: after 6 pushes into capacity 4, the buffer holds
    # markers 2,3,4,5 (0 and 1 were overwritten).
    assert sorted(buf.obs[:, 0].tolist()) == [2.0, 3.0, 4.0, 5.0]


def test_replay_buffer_sample_preserves_row_correspondence():
    buf = ReplayBuffer(capacity=10, obs_dim=1, n_actions=3, seed=0)
    masks = [
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ]
    for i, m in enumerate(masks):
        buf.push(obs=[float(i)], action=i, reward=float(i) * 10,
                  next_obs=[float(i) + 0.5], next_mask=m, done=False)
    batch = buf.sample(batch_size=50)
    for i in range(50):
        marker = int(batch["obs"][i, 0])
        assert batch["action"][i] == marker
        assert batch["reward"][i] == marker * 10
        assert batch["next_obs"][i, 0] == marker + 0.5
        assert list(batch["next_mask"][i]) == masks[marker]


def test_replay_buffer_sample_before_full_only_returns_pushed_rows():
    buf = ReplayBuffer(capacity=100, obs_dim=1, n_actions=2, seed=0)
    for i in range(3):
        buf.push(obs=[float(i)], action=0, reward=0.0,
                  next_obs=[0.0], next_mask=[True, True], done=False)
    batch = buf.sample(batch_size=20)
    assert set(batch["obs"][:, 0].tolist()) <= {0.0, 1.0, 2.0}
