import numpy as np
import torch
from geomloss import SamplesLoss
import torch.nn.functional as F

# This one only works for SMPL/STAR models
def edge_loss(pred, gt, edge_index):
    """
    Safe edge loss with index bounds checking.
    """
    B, V, _ = pred.shape  # B=batch size, V=vertices

    # Get edge vertex indices
    v1 = edge_index[0]
    v2 = edge_index[1]

    # Sanity check and filter invalid edges (v1 or v2 >= V)
    valid_mask = (v1 < V) & (v2 < V)
    v1 = v1[valid_mask]
    v2 = v2[valid_mask]

    if v1.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    # Compute edge vectors
    pred_edge_vectors = pred[:, v1, :] - pred[:, v2, :]  # [B, E, 3]
    gt_edge_vectors = gt[:, v1, :] - gt[:, v2, :]        # [B, E, 3]

    # Edge difference norm
    edge_diff = pred_edge_vectors - gt_edge_vectors
    edge_loss = torch.norm(edge_diff, dim=-1).mean()  # scalar

    return edge_loss


def kl_divergence(mu, logvar):
    """
    Computes KL divergence between N(mu, sigma^2) and N(0, 1)
    """
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()

def compute_mmd(z, prior_z, sigma=1.0):
    def gaussian_kernel(a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(0)
        diff = a - b
        return torch.exp(-torch.sum(diff ** 2, dim=2) / (2 * sigma ** 2))

    K_zz = gaussian_kernel(z, z)
    K_pp = gaussian_kernel(prior_z, prior_z)
    K_zp = gaussian_kernel(z, prior_z)

    mmd = K_zz.mean() + K_pp.mean() - 2 * K_zp.mean()
    return mmd


def reconstruction_loss(pred, target):
    return F.mse_loss(pred, target, reduction='mean')


def optimal_transport_loss(pred, target):
    """
    pred, target: [B, V, 3] point clouds
    Returns average Wasserstein loss per batch
    """
    w_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
    
    B = pred.shape[0]
    loss = 0.0
    for i in range(B):
        loss += w_loss(pred[i], target[i])
    return loss / B


def vae_loss(pred, target, mu, logvar, edge_index=None, loss_weights=None):
    """
    Computes total VAE loss with KL + reconstruction + OT

    Args:
        pred (Tensor): [B, V, 3] predicted vertices
        target (Tensor): [B, V, 3] ground truth
        mu, logvar: latent space parameters
        loss_weights: dict with keys "kl", "recon", "ot", "edge"
            and corresponding weights for each loss term

    Returns:
        total_loss, dict of individual losses
    """
    lw = loss_weights or {"kl": 1.0, "recon": 1.0, "ot": 0.0, "edge": 1.0}

    mmd = compute_mmd(mu, torch.zeros_like(mu), sigma=1.0)  # MMD loss
    kl = kl_divergence(mu, logvar)
    recon = reconstruction_loss(pred, target)
    ot = optimal_transport_loss(pred, target)
    edge = edge_loss(pred, target, edge_index) if edge_index is not None else 0.0

    total = lw["kl"] * mmd + lw["recon"] * recon + lw["ot"] * ot + lw["edge"] * edge

    return total, {"kl": mmd, "recon": recon, "ot": ot, "edge": edge}