import torch
import numpy as np

def evaluate_policy_batch(model, X, action, reward, device="cpu"):
    """
    Computes IPS & WIS for a single batch.
    Works even if dataset does NOT provide p (propensity score).
    """

    model.eval()
    with torch.no_grad():
        X = X.to(device)
        action = action.to(device)
        reward = reward.to(device)

        logits, _ = model(X)
        pi = torch.softmax(logits, dim=-1)

        # target probability of chosen action
        pi_a = pi[torch.arange(len(action)), action]

        # assume behavior policy was uniform random 1/K
        K = logits.size(1)
        p = torch.full_like(pi_a, 1.0 / K)

        w = (pi_a / p).cpu().numpy()
        ips = w * reward.cpu().numpy()

        ips_est = np.mean(ips)
        wis_est = np.sum(ips) / np.sum(w)

        return ips_est, wis_est









