from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Conditional GAN generator for tabular data.

    Input:
        z : (batch, z_dim)         random noise
        c : (batch, c_dim)         one-hot condition / class label

    Output:
        x_fake : (batch, x_dim)    synthetic feature vector
    """

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        z_dim: int = 16,
        hidden: int = 128,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden, x_dim)
            # no activation:
            # output is in standardized feature space,
            # so raw linear output is fine
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, c], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    """
    Conditional GAN discriminator for tabular data.

    Input:
        x : (batch, x_dim) real or fake feature vector
        c : (batch, c_dim) one-hot condition / class label

    Output:
        logit : (batch, 1) real/fake score (before sigmoid)
    """

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim

        self.net = nn.Sequential(
            nn.Linear(x_dim + c_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden, 1)
            # no sigmoid here:
            # use BCEWithLogitsLoss in training
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        xc = torch.cat([x, c], dim=1)
        return self.net(xc)


class CGAN(nn.Module):
    """
    Convenience wrapper holding both generator and discriminator.
    """

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        z_dim: int = 16,
        hidden: int = 128,
        d_dropout: float = 0.1,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.generator = Generator(
            x_dim=x_dim,
            c_dim=c_dim,
            z_dim=z_dim,
            hidden=hidden,
        )

        self.discriminator = Discriminator(
            x_dim=x_dim,
            c_dim=c_dim,
            hidden=hidden,
            dropout=d_dropout,
        )

    def sample(self, n: int, c: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Sample fake data from the generator.

        Args:
            n: batch size
            c: one-hot condition tensor of shape (n, c_dim)
            device: torch device

        Returns:
            x_fake: (n, x_dim)
        """
        z = torch.randn(n, self.z_dim, device=device)
        return self.generator(z, c)


def weights_init(m: nn.Module) -> None:
    """
    Optional GAN-style weight initialization.
    Call:
        model.apply(weights_init)
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)