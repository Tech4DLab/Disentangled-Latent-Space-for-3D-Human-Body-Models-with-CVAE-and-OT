import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

"""
VAE loss function combining reconstruction loss, KL divergence, and optional edge loss.
args:
    pred: predicted mesh vertices [B, V, 3]
    target: ground truth mesh vertices [B, V, 3]
    mu: mean of the latent space
    logvar: log variance of the latent space
    edge_index: edge index for edge loss (optional)
    loss_weights: dictionary with weights for KL, reconstruction, and OT losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool

class Encoder(nn.Module):
    def __init__(self, in_channels=3, cond_dim=3, hidden_channels=128, latent_dim=16, K=3):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Graph convolution layers
        self.conv1 = ChebConv(in_channels + cond_dim, hidden_channels, K)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = ChebConv(hidden_channels, hidden_channels, K)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # Fully connected layers to produce latent parameters
        self.fc_mu = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels, latent_dim)

        # Disentangled attribute predictors
        self.fc_weight = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        self.fc_height = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        self.fc_gender = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index, batch, y):
        # Expand y to match x (node-level)
        cond = y[batch]  # [N, cond_dim]
        x = torch.cat([x, cond], dim=1)  # [N, 3 + cond_dim]

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)  # [B, hidden_channels]

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.sampling(mu, logvar)

        z_weight = z[:, :6]
        z_height = z[:, 6:12]
        z_gender = z[:, 12:]

        pred_weight = self.fc_weight(z_weight)
        pred_height = self.fc_height(z_height)
        pred_gender = self.fc_gender(z_gender)

        return mu, logvar, z, (pred_weight, pred_height, pred_gender)


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, cond_dim=3, num_points=6890):
        super(Decoder, self).__init__()
        hidden_dim = 512
        self.num_points = num_points

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * 3)
        )

    def forward(self, z, y):
        zy = torch.cat([z, y], dim=1)  # [B, latent + cond]
        out = self.decoder(zy)        # [B, num_points * 3]
        return out.view(-1, self.num_points, 3)