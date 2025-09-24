import torch
import torch.optim as optim
from lib.losses import vae_loss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import os

def train_epoch(encoder, decoder, dataloader, optimizer, device, loss_weights, gamma=1.0, kl_weight=1.0):
    encoder.train()
    decoder.train()

    # Initialize accumulators
    total_loss = kl_total = recon_total = ot_total = edge_total = 0.0
    weight_total = height_total = gender_total = 0.0

    for batch in dataloader:
        batch = batch.to(device)
        y = batch.y.to(device)  # [B, 3] → weight_norm, height_norm, gender

        # Forward
        mu, logvar, z, (pred_weight, pred_height, pred_gender) = encoder(batch.x, batch.edge_index, batch.batch, y)
        pred_verts = decoder(z, y)

        # Vertex ground truth
        B = batch.num_graphs
        V = pred_verts.shape[1]
        gt_verts = batch.x.view(B, V, 3)

        # VAE losses
        _, losses_dict = vae_loss(pred_verts, gt_verts, mu, logvar, batch.edge_index, loss_weights)
        loss_vae = (
            loss_weights["recon"] * losses_dict["recon"] +
            kl_weight * loss_weights["kl"] * losses_dict["kl"] +
            loss_weights["ot"] * losses_dict["ot"] +
            loss_weights["edge"] * losses_dict["edge"]
        )

        # Supervised losses
        loss_weight = F.mse_loss(pred_weight.squeeze(), y[:, 0])
        loss_height = F.mse_loss(pred_height.squeeze(), y[:, 1])
        pos_weight = torch.tensor([3.0], device=device)
        loss_gender = F.binary_cross_entropy_with_logits(pred_gender.squeeze(), y[:, 2], pos_weight=pos_weight)

        loss_supervised = gamma * (loss_weight + loss_height + loss_gender)
        loss_total = loss_vae + loss_supervised

        # Backpropagation
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss_total.item()
        kl_total += losses_dict["kl"].item()
        recon_total += losses_dict["recon"].item()
        ot_total += losses_dict["ot"].item()
        edge_total += losses_dict["edge"].item()
        weight_total += loss_weight.item()
        height_total += loss_height.item()
        gender_total += loss_gender.item()

    avg = lambda x: x / len(dataloader)
    return {
        "total": avg(total_loss),
        "kl": avg(kl_total),
        "recon": avg(recon_total),
        "ot": avg(ot_total),
        "edge": avg(edge_total),
        "weight": avg(weight_total),
        "height": avg(height_total),
        "gender": avg(gender_total),
    }

def train_model(encoder, decoder, dataloader, num_epochs, device, loss_weights, warmup_epochs=20):
    # Optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-3
    )

    loss_history = {key: [] for key in ["total", "kl", "recon", "ot", "edge", "weight", "height", "gender"]}

    # Training loop
    for epoch in range(num_epochs):
        kl_weight = min(1.0, epoch / warmup_epochs)

        epoch_losses = train_epoch(
            encoder, decoder, dataloader,
            optimizer, device, loss_weights, kl_weight=kl_weight
        )

        for key in loss_history:
            loss_history[key].append(epoch_losses[key])

        print(f"Epoch {epoch + 1}/{num_epochs} | " +
              " | ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]))

    # Plot
    for key, values in loss_history.items():
        plt.plot(values, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("losses_plot.png")
    plt.show()

    # Save CSV
    csv_filename = "Results/loss_history.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["epoch", "run_name"] + list(loss_history.keys()))

        run_name = "CVAE Disentangled (gamma=0.5) + KL Warmup"
        for i in range(num_epochs):
            row = [i + 1, run_name] + [loss_history[key][i] for key in loss_history]
            writer.writerow(row)

    return encoder, decoder