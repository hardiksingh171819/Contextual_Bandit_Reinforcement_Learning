# src/model.py
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_actions):
        super().__init__()

        layers = []
        last_dim = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h

        self.backbone = nn.Sequential(*layers)
        self.logits = nn.Linear(last_dim, num_actions)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.logits(z)
        return logits, z


