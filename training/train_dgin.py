# training/train_dgin.py
from models.dgin.encoder import DGINEncoder
from models.dgin.policy import DGINPolicy
from utils.dataloader import load_training_data
from training.logger import Logger

import torch
import torch.optim as optim
import yaml

def train_dgin(config_path="training/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    encoder = DGINEncoder(config["encoder"])
    policy = DGINPolicy(config["policy"]["latent_dim"])
    logger = Logger()

    # Load data
    train_loader = load_training_data(config["data"])

    optimizer = optim.Adam(list(encoder.parameters()) + list(policy.parameters()), lr=config["training"]["lr"])

    for epoch in range(config["training"]["epochs"]):
        for batch in train_loader:
            # Forward pass
            latent = encoder.forward(batch["player_states"], batch["game_context"])
            actions = policy.get_action(latent)

            # Compute loss (stub)
            loss = torch.tensor(0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.log(f"Epoch {epoch+1} - Loss: {loss.item()}")

if __name__ == "__main__":
    train_dgin()
