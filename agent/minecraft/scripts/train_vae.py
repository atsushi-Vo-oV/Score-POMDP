"""Train a small convolutional VAE on collected 128x128 SBWalk frames.
Usage: train_vae.py <out.pt> data/frames_*.npz [--latent 32] [--epochs 30] [--beta 1.0] [--device cuda]
Writes the encoder/decoder weights plus normalisation, and a reconstruction sheet (PNG) next to the checkpoint.
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent: int):
        super().__init__()
        ch = [3, 32, 64, 128, 256]
        self.conv = nn.Sequential(*[m for i in range(4) for m in (nn.Conv2d(ch[i], ch[i + 1], 4, 2, 1), nn.SiLU())])  # 128 -> 8
        self.fc = nn.Linear(256 * 8 * 8, 2 * latent)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        mu, log_var = self.fc(h).chunk(2, -1)
        return mu, log_var.clamp(-8, 4)


class Decoder(nn.Module):
    def __init__(self, latent: int):
        super().__init__()
        self.fc = nn.Linear(latent, 256 * 8 * 8)
        ch = [256, 128, 64, 32]
        layers = [m for i in range(3) for m in (nn.ConvTranspose2d(ch[i], ch[i + 1], 4, 2, 1), nn.SiLU())]
        layers.append(nn.ConvTranspose2d(32, 3, 4, 2, 1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        return self.deconv(F.silu(self.fc(z)).view(-1, 256, 8, 8))


class VAE(nn.Module):
    def __init__(self, latent: int):
        super().__init__()
        self.latent = latent
        self.encoder, self.decoder = Encoder(latent), Decoder(latent)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = mu + (0.5 * log_var).exp() * torch.randn_like(mu)
        return self.decoder(z), mu, log_var


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    files = sorted(f for pat in a.inputs for f in glob.glob(pat))
    frames = np.concatenate([np.load(f)["frames"] for f in files])
    print("frames", frames.shape, "from", files, flush=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(frames))
    n_val = max(256, len(frames) // 20)
    val, train = perm[:n_val], perm[n_val:]
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = VAE(a.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    x_all = torch.as_tensor(frames).permute(0, 3, 1, 2)  # uint8 NCHW on CPU

    def batch(idx):
        return x_all[idx].to(device).float() / 255.0

    t0 = time.time()
    for epoch in range(a.epochs):
        model.train()
        rng.shuffle(train)
        tot = rec_tot = kl_tot = 0.0
        for i in range(0, len(train), a.batch):
            x = batch(train[i:i + a.batch])
            recon, mu, log_var = model(x)
            rec = F.mse_loss(recon, x, reduction="sum") / len(x)
            kl = -0.5 * (1 + log_var - mu.square() - log_var.exp()).sum(-1).mean()
            loss = rec + a.beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            rec_tot += rec.item() * len(x)
            kl_tot += kl.item() * len(x)
        model.eval()
        with torch.no_grad():
            xv = batch(val)
            recon, mu, log_var = model(xv)
            v_rec = F.mse_loss(recon, xv, reduction="sum").item() / len(xv)
            psnr = 10 * np.log10(1.0 / F.mse_loss(recon, xv).item())
        print(f"epoch {epoch + 1}/{a.epochs} loss {tot / len(train):.1f} rec {rec_tot / len(train):.1f} kl {kl_tot / len(train):.1f} | val rec {v_rec:.1f} psnr {psnr:.1f} dB | {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        mus = torch.cat([model.encoder(batch(train[i:i + 512]))[0] for i in range(0, min(len(train), 8192), 512)])
    torch.save({"latent": a.latent, "model": model.state_dict(), "mu_mean": mus.mean(0).cpu(), "mu_std": mus.std(0).cpu(),
                "frames": int(len(frames)), "epochs": a.epochs, "beta": a.beta}, a.out)
    # reconstruction sheet: 8 validation frames, originals above reconstructions
    from PIL import Image
    with torch.no_grad():
        xv = batch(val[:8])
        recon = model.decoder(model.encoder(xv)[0]).clamp(0, 1)
    sheet = torch.cat([xv, recon], 2).permute(0, 2, 3, 1).reshape(2 * 128, 8 * 128, 3)  # rows: orig | recon, 8 columns
    sheet = torch.cat([torch.cat([xv[i], recon[i]], 1) for i in range(8)], 2).permute(1, 2, 0)
    Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8)).save(a.out.replace(".pt", "_recon.png"))
    print("saved", a.out, "latent std mean", float(mus.std(0).mean()))


if __name__ == "__main__":
    main()
