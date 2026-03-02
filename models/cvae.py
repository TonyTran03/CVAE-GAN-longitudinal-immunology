# models/cvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    """
    Conditional VAE for tabular features.

    Encoder: [x, c] -> (mu, logvar)
    Reparam: z = mu + exp(0.5*logvar) * eps
    Decoder: [z, c] -> x_hat
    """
    def __init__(self, x_dim: int, c_dim: int, z_dim: int, hidden: int):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + c_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + c_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),  # linear output for MSE/Gaussian recon
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        xc = torch.cat([x, c], dim=1)
        h = self.enc(xc)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        eps = torch.randn_like(mu)
        sigma = torch.exp(0.5 * logvar)
        return mu + sigma * eps

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        zc = torch.cat([z, c], dim=1)
        return self.dec(zc)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar
    
if __name__ == "__main__":
    # Quick test
    x_dim, c_dim, z_dim, hidden = 10, 2, 4, 16
    model = CVAE(x_dim, c_dim, z_dim, hidden)
    x = torch.randn(5, x_dim)
    c = F.one_hot(torch.randint(0, c_dim, (5,)), num_classes=c_dim).float()
    x_hat, mu, logvar = model(x, c)
    print("x_hat shape:", x_hat.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)