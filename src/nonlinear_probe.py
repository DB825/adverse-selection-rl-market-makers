"""Predeclared nonlinear recent-history decoder with validation-only selection."""
import copy
import hashlib

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


def fit_history_decoder(x_train, y_train, x_validation, y_validation, x_test, seed,
                        epochs=40, every=5, weight_decays=(.0001, .01), batch_size=512):
    if epochs < every or epochs % every:
        raise ValueError('Epoch budget must be a positive multiple of validation interval')
    arrays = [np.asarray(a, dtype=np.float64) for a in (x_train, y_train, x_validation, y_validation, x_test)]
    if any(a.ndim != 2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Decoder inputs must be finite matrices')
    x_train, y_train, x_validation, y_validation, x_test = arrays
    if (len(x_train) != len(y_train) or len(x_validation) != len(y_validation)
            or y_train.shape[1] != 6 or y_validation.shape[1] != 6):
        raise ValueError('Require aligned observations and six targets')
    scaler = StandardScaler().fit(x_train)
    center = y_train.mean(0)
    scale = np.maximum(y_train.std(0), 1e-8)
    xt = torch.tensor(scaler.transform(x_train), dtype=torch.float32)
    xv = torch.tensor(scaler.transform(x_validation), dtype=torch.float32)
    yt = torch.tensor((y_train-center)/scale, dtype=torch.float32)
    yv = torch.tensor((y_validation-center)/scale, dtype=torch.float32)
    torch.set_num_threads(1)
    best, trials = None, []
    with torch.random.fork_rng(devices=[]):
        for decay in weight_decays:
            torch.manual_seed(seed)
            net = nn.Sequential(nn.Linear(xt.shape[1], 64), nn.Tanh(), nn.Linear(64, 64),
                                nn.Tanh(), nn.Linear(64, 6))
            optimizer = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=decay)
            order = np.random.default_rng(seed)
            for epoch in range(1, epochs+1):
                net.train()
                indices = order.permutation(len(xt))
                for start in range(0, len(indices), batch_size):
                    batch = indices[start:start+batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = ((net(xt[batch])-yt[batch])**2).mean()
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite decoder training loss')
                    loss.backward(); optimizer.step()
                if epoch % every == 0:
                    net.eval()
                    with torch.no_grad():
                        score = float(((net(xv)-yv)**2).mean())
                    if not np.isfinite(score):
                        raise ValueError('Nonfinite validation score')
                    trials.append({'weight_decay': decay, 'epoch': epoch, 'validation_scaled_mse': score})
                    if best is None or score < best['validation_scaled_mse']:
                        best = dict(trials[-1], state=copy.deepcopy(net.state_dict()))
        net.load_state_dict(best['state']); net.eval()
        with torch.no_grad():
            prediction = net(torch.tensor(scaler.transform(x_test), dtype=torch.float32)).numpy().astype(float)*scale+center
    state_hash = hashlib.sha256(b''.join(t.numpy().tobytes() for t in best['state'].values())).hexdigest()
    metadata = {k: v for k, v in best.items() if k != 'state'}
    metadata.update(seed=seed, epochs_per_candidate=epochs, validation_every=every, trials=trials,
                    state_sha256=state_hash, parameter_count=sum(p.numel() for p in net.parameters()),
                    input_scaler_mean=scaler.mean_.tolist(), input_scaler_scale=scaler.scale_.tolist(),
                    target_mean=center.tolist(), target_scale=scale.tolist(),
                    architecture=[x_train.shape[1], 64, 64, 6], test_used_for_selection=False)
    return prediction, metadata, best['state']
