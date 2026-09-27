"""Equivalent fused LSTM reset path for padded sb3-contrib PPO sequences.

SB3 separates training sequences at episode boundaries. Its stock helper still
loops over individual time steps when *any* reset is present. When resets only
occur at the first time step, resetting h,c once and processing the sequence in
one PyTorch LSTM call is equivalent (up to floating point rounding). Interior
resets fall back to the unmodified upstream implementation. No PPO update,
loss, sequence sampling, padding, or gradient truncation rule is changed.

This small policy subclass is tested against installed sb3-contrib 2.7.0 for
outputs, final h/c, input/state gradients and every LSTM parameter gradient.
It requires the default dropout=0; dropout-enabled LSTMs use the stock path.
"""

import torch
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


class FusedResetRecurrentPolicy(RecurrentActorCriticPolicy):
    @staticmethod
    def _process_sequence(features, lstm_states, episode_starts, lstm):
        n_seq = lstm_states[0].shape[1]
        resets = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        if lstm.dropout != 0 or not torch.all(resets[1:] == 0):
            return RecurrentActorCriticPolicy._process_sequence(
                features, lstm_states, episode_starts, lstm,
            )
        sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        mask = (1.0 - resets[0]).view(1, n_seq, 1)
        output, outgoing = lstm(sequence, (mask * lstm_states[0], mask * lstm_states[1]))
        return torch.flatten(output.transpose(0, 1), start_dim=0, end_dim=1), outgoing
