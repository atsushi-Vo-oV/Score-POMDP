"""Write an untrained VAE checkpoint in the train_vae.py format (for plumbing tests of the minerl backend)."""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/scripts")
from train_vae import VAE  # noqa: E402

out, latent = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32
torch.manual_seed(0)
model = VAE(latent)
torch.save({"latent": latent, "model": model.state_dict(), "mu_mean": torch.zeros(latent), "mu_std": torch.ones(latent),
            "frames": 0, "epochs": 0, "beta": 1.0}, out)
print("wrote", out)
