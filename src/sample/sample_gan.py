from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from models.gan import CGAN
from util.config import Config
from util.transformation import make_transform


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    cfg = Config(**ckpt["cfg"])

    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)

    x_dim = scaler_mean.shape[0]

    model = CGAN(
        x_dim=x_dim,
        c_dim=2,
        z_dim=cfg.z_dim,
        hidden=cfg.hidden,
        d_dropout=0.1,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = make_transform(cfg.x_transform)

    return model, cfg, scaler_mean, scaler_scale, transform


@torch.no_grad()
def sample_class(model: CGAN, n: int, y_label: int,
                 scaler_mean: np.ndarray,
                 scaler_scale: np.ndarray,
                 transform,
                 device: torch.device) -> np.ndarray:
    model = model.to(device)
    model.eval()

    c = F.one_hot(
        torch.full((n,), y_label, dtype=torch.long, device=device),
        num_classes=2
    ).float()

    z = torch.randn(n, model.z_dim, device=device)

    # generator outputs standardized-space samples
    x_scaled = model.generator(z, c).cpu().numpy()

    # inverse standardization -> transformed space
    x_t = x_scaled * scaler_scale + scaler_mean

    # inverse transform -> original feature space
    x = transform.inverse(x_t)

    return x.astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt_path = Path("data/log1p/gan_best_xlog1p_z16_h128_lr0p001_seed42.pt")
    model, cfg, mean, scale, transform = load_checkpoint(ckpt_path)

    cfg.ensure_dirs()

    n = 91

    X_y0 = sample_class(model, n, 0, mean, scale, transform, device)
    X_y1 = sample_class(model, n, 1, mean, scale, transform, device)

    out_dir = cfg.output_path / (cfg.x_transform or "none")
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    out0 = sample_dir / f"gan_synth_seed{cfg.seed}_y0.npz"
    out1 = sample_dir / f"gan_synth_seed{cfg.seed}_y1.npz"

    np.savez(out0, X=X_y0)
    np.savez(out1, X=X_y1)

    print("Saved:")
    print(" ", out0, X_y0.shape)
    print(" ", out1, X_y1.shape)


if __name__ == "__main__":
    main()