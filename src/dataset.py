# src/dataset.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class BanditDataset(Dataset):
    """
    Converts the Avazu CTR dataset into a contextual bandit dataset:

    reward = click
    action = banner_pos
    features = all remaining numeric-encoded columns
    """

    def __init__(self, csv_path, nrows=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path, nrows=nrows)
        print(f"[INFO] Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

        # Reward (0 or 1)
        self.rewards = torch.tensor(df["click"].astype(float).values, dtype=torch.float32)

        # Action = banner_pos
        if "banner_pos" not in df.columns:
            raise KeyError("'banner_pos' column not found — cannot define actions.")

        self.actions = torch.tensor(df["banner_pos"].astype(int).values, dtype=torch.long)

        # Remove non-feature columns
        exclude = {"id", "click", "banner_pos"}

        # convert all remaining categorical columns using label encoding
        feature_cols = [c for c in df.columns if c not in exclude]

        # Convert every column to numeric using category codes
        for col in feature_cols:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category").cat.codes

        self.features = torch.tensor(df[feature_cols].astype(float).values, dtype=torch.float32)

        print(f"[DEBUG] Using {len(feature_cols)} feature columns: {feature_cols[:10]}")
        print(f"[DEBUG] Feature tensor shape: {self.features.shape}")

        self.feature_dim = len(feature_cols)

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        return self.features[idx], self.actions[idx], self.rewards[idx]
