import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from src.dataset import BanditDataset
from src.model import PolicyNet
from src.utils import load_config, get_device
from src.eval import evaluate_policy_batch

# Load config & device
config = load_config("configs/config.yaml")
device = get_device(config["train"]["device"])

# Dataset + Dataloader
dataset = BanditDataset(config["data"]["path"], config["data"]["nrows"])
loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)

# Model setup
input_dim = dataset.feature_dim
num_actions = config["model"]["num_actions"]

model = PolicyNet(input_dim, config["model"]["hidden_sizes"], num_actions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
loss_fn = torch.nn.CrossEntropyLoss()

losses = []
ips_history = []
wis_history = []

# ============================
# TRAINING LOOP
# ============================
for epoch in range(1, config["train"]["epochs"] + 1):

    loop = tqdm(loader, desc=f"Epoch {epoch}/{config['train']['epochs']}")
    epoch_losses = []
    epoch_ips = []
    epoch_wis = []

    for X, a, r in loop:
        X, a, r = X.to(device), a.to(device), r.to(device)

        logits, _ = model(X)
        loss = loss_fn(logits, a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

        # compute IPS & WIS for this batch
        ips, wis = evaluate_policy_batch(model, X, a, r, device)
        epoch_ips.append(ips)
        epoch_wis.append(wis)

        loop.set_postfix(loss=loss.item(), IPS=ips, WIS=wis)

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    avg_ips = sum(epoch_ips) / len(epoch_ips)
    avg_wis = sum(epoch_wis) / len(epoch_wis)

    losses.append(avg_loss)
    ips_history.append(avg_ips)
    wis_history.append(avg_wis)

    print(f"\nEpoch {epoch}: Loss={avg_loss:.4f} | IPS={avg_ips:.6f} | WIS={avg_wis:.6f}\n")


# ============================
# SAVE MODEL & PLOTS
# ============================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/policy.pt")
print("model saved")

# Loss curve
plt.figure()
plt.plot(losses)
plt.title("Training Loss Curve")
plt.savefig("logs/loss.png")

# IPS curve
plt.figure()
plt.plot(ips_history)
plt.title("IPS per Epoch")
plt.savefig("logs/ips_curve.png")

# WIS curve
plt.figure()
plt.plot(wis_history)
plt.title("WIS per Epoch")
plt.savefig("logs/wis_curve.png")

print("loss + IPS + WIS plots saved")


