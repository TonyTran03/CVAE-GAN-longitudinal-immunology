from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.config import Config
from src.data import load_rdata_xy, make_loaders
from util.utils import set_seed
from models.gan import CGAN, weights_init


cfg = Config()
cfg.ensure_dirs()
set_seed(cfg.seed)


def run_tag(cfg: Config) -> str:
    lr = str(cfg.lr).replace(".", "p")
    return f"x{cfg.x_transform}_z{cfg.z_dim}_h{cfg.hidden}_lr{lr}_seed{cfg.seed}"


@torch.no_grad()
def evaluate_discriminator(model: CGAN, loader, device) -> Dict[str, float]:
    """
    Evaluate discriminator loss/accuracy on a validation loader.
    This is not a final synthetic-quality metric, just a training diagnostic.
    """
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x_real, c in loader:
        x_real, c = x_real.to(device), c.to(device)
        batch_size = x_real.size(0)

        real_targets = torch.ones(batch_size, 1, device=device)
        fake_targets = torch.zeros(batch_size, 1, device=device)

        # real
        real_logits = model.discriminator(x_real, c)
        real_loss = bce(real_logits, real_targets)

        # fake
        z = torch.randn(batch_size, model.z_dim, device=device)
        x_fake = model.generator(z, c)
        fake_logits = model.discriminator(x_fake, c)
        fake_loss = bce(fake_logits, fake_targets)

        loss = real_loss + fake_loss

        # rough discriminator accuracy
        real_pred = (torch.sigmoid(real_logits) >= 0.5).float()
        fake_pred = (torch.sigmoid(fake_logits) < 0.5).float()

        acc = torch.cat([real_pred, fake_pred], dim=0).mean().item()

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return {
        "d_loss": total_loss / n_batches,
        "d_acc": total_acc / n_batches,
    }


def main():
    tag = run_tag(cfg)
    best_path = cfg.out_dir / f"gan_best_{tag}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Transform:", cfg.x_transform)

    X, y = load_rdata_xy(cfg.data_path, x_key=cfg.x_key, y_key=cfg.y_key)

    # keep same preprocessing / split pipeline as CVAE
    train_loader, val_loader, scaler = make_loaders(
        X,
        y,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        x_transform=cfg.x_transform,
    )

    x_dim = X.shape[1]
    c_dim = 2

    model = CGAN(
        x_dim=x_dim,
        c_dim=c_dim,
        z_dim=cfg.z_dim,
        hidden=cfg.hidden,
        d_dropout=0.1,
    ).to(device)

    model.apply(weights_init)

    opt_g = torch.optim.Adam(model.generator.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        g_running = 0.0
        d_running = 0.0
        n_batches = 0

        for x_real, c in train_loader:
            x_real, c = x_real.to(device), c.to(device)
            batch_size = x_real.size(0)

            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            # -------------------------
            # 1) Train discriminator
            # -------------------------
            opt_d.zero_grad(set_to_none=True)

            # real loss
            real_logits = model.discriminator(x_real, c)
            d_loss_real = bce(real_logits, real_targets)

            # fake loss
            z = torch.randn(batch_size, model.z_dim, device=device)
            x_fake = model.generator(z, c)
            fake_logits = model.discriminator(x_fake.detach(), c)
            d_loss_fake = bce(fake_logits, fake_targets)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # -------------------------
            # 2) Train generator
            # -------------------------
            opt_g.zero_grad(set_to_none=True)

            z = torch.randn(batch_size, model.z_dim, device=device)
            x_fake = model.generator(z, c)
            fake_logits = model.discriminator(x_fake, c)

            # generator wants discriminator to call fake samples "real"
            g_loss = bce(fake_logits, real_targets)
            g_loss.backward()
            opt_g.step()

            d_running += d_loss.item()
            g_running += g_loss.item()
            n_batches += 1

        train_metrics = {
            "d_loss": d_running / n_batches,
            "g_loss": g_running / n_batches,
        }

        val_metrics = evaluate_discriminator(model, val_loader, device)

        # crude checkpoint rule:
        # if discriminator is less dominant / more balanced, that is generally better
        # use validation discriminator loss as a simple tracking metric
        if val_metrics["d_loss"] < best_val:
            best_val = val_metrics["d_loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "cfg": cfg.__dict__,
                    "x_transform": cfg.x_transform,
                },
                best_path,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train d_loss={train_metrics['d_loss']:.4f} "
                f"g_loss={train_metrics['g_loss']:.4f} | "
                f"val d_loss={val_metrics['d_loss']:.4f} "
                f"d_acc={val_metrics['d_acc']:.4f}"
            )

    print("Training complete.")
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()