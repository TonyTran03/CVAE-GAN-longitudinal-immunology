# train_cvae.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from util.config import Config
from src.data import load_rdata_xy, make_loaders
from util.utils import set_seed

from models.cvae import CVAE
cfg = Config()
cfg.ensure_dirs()
set_seed(cfg.seed)
def elbo_loss(x, x_hat, mu, logvar, beta: float):
    # Recon: MSE sum over features, mean over batch
    recon = ((x_hat - x) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1.0 + logvar - mu**2 - torch.exp(logvar)).sum(dim=1)).mean()
    total = recon + beta * kl
    return total, recon, kl

def run_tag(cfg: Config) -> str:
    noise = str(cfg.decoder_noise).replace(".", "p")
    beta  = str(cfg.beta).replace(".", "p")
    lr    = str(cfg.lr).replace(".", "p")
    return f"x{cfg.x_transform}_z{cfg.z_dim}_h{cfg.hidden}_b{beta}_lr{lr}_dn{noise}_seed{cfg.seed}"


@torch.no_grad()
def evaluate(model: CVAE, loader, device, beta: float, decoder_noise: float = 0.0) -> Dict[str, float]:
    model.eval()
    tot = rec = kl = 0.0
    n = 0
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        x_hat, mu, logvar = model(x, c)

        if decoder_noise > 0:
            x_hat = x_hat + decoder_noise * torch.randn_like(x_hat)

        loss, r, k = elbo_loss(x, x_hat, mu, logvar, beta=beta)
        tot += loss.item()
        rec += r.item()
        kl += k.item()
        n += 1

    return {"loss": tot / n, "recon": rec / n, "kl": kl / n}

def main():

    tag = run_tag(cfg)
    best_path = cfg.out_dir/ f"cvae_best_{tag}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", cfg.x_transform)
    X, y = load_rdata_xy(cfg.data_path, x_key=cfg.x_key, y_key=cfg.y_key)
    feature_names = [f"x{i}" for i in range(X.shape[1])]
    train_loader, val_loader, scaler = make_loaders(
        X, y,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        x_transform=cfg.x_transform
    )

    x_dim = X.shape[1]
    c_dim = 2

    model = CVAE(x_dim=x_dim, c_dim=c_dim, z_dim=cfg.z_dim, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot = rec = kl = 0.0
        n = 0
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            opt.zero_grad(set_to_none=True)

            x_hat, mu, logvar = model(x, c)
            if cfg.decoder_noise > 0:
                x_hat = x_hat + cfg.decoder_noise * torch.randn_like(x_hat)

            loss, r, k = elbo_loss(x, x_hat, mu, logvar, beta=cfg.beta)

            loss.backward()
            opt.step()

            tot += loss.item()
            rec += r.item()
            kl += k.item()
            n += 1

        train_metrics = {"loss": tot / n, "recon": rec / n, "kl": kl / n}
        val_metrics = evaluate(model, val_loader, device, beta=cfg.beta)


        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "feature_names": feature_names,
                    "cfg": cfg.__dict__,
                    "x_transform": cfg.x_transform,
                },
                best_path,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train loss={train_metrics['loss']:.4f} recon={train_metrics['recon']:.4f} kl={train_metrics['kl']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} recon={val_metrics['recon']:.4f} kl={val_metrics['kl']:.4f}"
            )

    print("didn't crash")


if __name__ == "__main__":
    main()