import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import torch
import torch.nn.functional as F 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
import trimesh 
from sklearn.metrics import silhouette_score

# Computes Chamfer Distance between two point clouds.
def chamfer_distance(p1, p2):
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    dists1, _ = tree1.query(p2)
    dists2, _ = tree2.query(p1)
    return np.mean(dists1**2) + np.mean(dists2**2)

# Computes Chamfer Distance between two point clouds.
def evaluate_and_visualize(encoder, decoder, dataloader, device, scene_bool=False, num_samples=5,
                            save_csv_path="errores_por_sample.csv", save_global_metrics=True):

    encoder.eval()
    decoder.eval()

    total_vertex_error = total_chamfer = total_wasserstein = 0.0
    total_samples = samples_visualized = 0

    w_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
    error_data = []
    props_error_list = []

    # Para separar por gender
    vertex_errors = {0: [], 1: []}
    chamfer_errors = {0: [], 1: []}
    wasserstein_errors = {0: [], 1: []}

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            y = batch.y.to(device)

            mu, logvar, z, _= encoder(batch.x, batch.edge_index, batch.batch, y)
            pred_verts = decoder(z, y)

            B, V = pred_verts.shape[:2]
            gt_verts = batch.x.view(B, V, 3)
            mean_error = torch.norm(pred_verts - gt_verts, dim=-1).mean(dim=1)

            ids = batch.id.cpu().numpy() if hasattr(batch, "id") else [None] * B
            sesiones = batch.sesion.cpu().numpy() if hasattr(batch, "sesion") else [None] * B
            genders = batch.gender.cpu().numpy() if hasattr(batch, "gender") else [-1] * B

            for i in range(B):
                pred = pred_verts[i].detach().cpu().numpy()
                gt = gt_verts[i].detach().cpu().numpy()

                chamfer = chamfer_distance(gt, pred)
                wasserstein = w_loss(
                    torch.tensor(pred, device=device),
                    torch.tensor(gt, device=device)
                ).item()
                v_error = mean_error[i].item()

                error_data.append({
                    "id": ids[i],
                    "sesion": sesiones[i],
                    "gender": genders[i],
                    "vertex_error": v_error,
                    "chamfer_error": chamfer,
                    "wasserstein_error": wasserstein
                })
                if scene_bool and samples_visualized < num_samples:
                    try:
                        pc_pred = trimesh.PointCloud(pred, colors=[255, 0, 0, 255])
                        pc_gt = trimesh.PointCloud(gt, colors=[0, 255, 0, 255])
                        trimesh.Scene([pc_gt, pc_pred]).show()
                        samples_visualized += 1
                    except Exception as e:
                        print(f"[Scene Error] {e}")

                total_vertex_error += v_error
                total_chamfer += chamfer
                total_wasserstein += wasserstein
                total_samples += 1

                if genders[i] in [0, 1]:
                    vertex_errors[genders[i]].append(v_error)
                    chamfer_errors[genders[i]].append(chamfer)
                    wasserstein_errors[genders[i]].append(wasserstein)

    pd.DataFrame(error_data).to_csv(save_csv_path, index=False)
    pd.DataFrame(props_error_list).to_csv(save_csv_path.replace(".csv", "_props.csv"), index=False)

    if save_global_metrics:
        resumen = [{
            "grupo": "Global",
            "samples": total_samples,
            "vertex_error": total_vertex_error / total_samples,
            "chamfer_error": total_chamfer / total_samples,
            "wasserstein_error": total_wasserstein / total_samples
        }]

        for g, name in zip([0, 1], ["woman", "man"]):
            if vertex_errors[g]:
                resumen.append({
                    "grupo": name,
                    "samples": len(vertex_errors[g]),
                    "vertex_error": np.mean(vertex_errors[g]),
                    "chamfer_error": np.mean(chamfer_errors[g]),
                    "wasserstein_error": np.mean(wasserstein_errors[g])
                })

        pd.DataFrame(resumen).to_csv(save_csv_path.replace(".csv", "_resumen.csv"), index=False)

    print(f"\n==== Global metrics ({total_samples} samples) ====")
    print(f"Vertex Error:     {total_vertex_error / total_samples:.4f}")
    print(f"Chamfer Distance: {total_chamfer / total_samples:.4f}")
    print(f"Wasserstein Dist: {total_wasserstein / total_samples:.4f}")
    print(f"MSE distance: {np.mean(np.array([e['vertex_error'] for e in error_data]) ** 2):.4f} mm²")

    for g, name in zip([0, 1], ["woman", "man"]):
        if vertex_errors[g]:
            print(f"\n--- {name} ({len(vertex_errors[g])} samples) ---")
            print(f"Vertex Error:     {np.mean(vertex_errors[g]):.4f}")
            print(f"Chamfer Distance: {np.mean(chamfer_errors[g]):.4f}")
            print(f"Wasserstein Dist: {np.mean(wasserstein_errors[g]):.4f}")


# Extracts latent vectors and conditioning variables from dataloader.
def extract_latents_and_variables(encoder, dataloader, device):
    encoder.eval()
    latents, y_all, ids, sesiones = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            y = batch.y.to(device)
            mu, logvar, z, _ = encoder(batch.x, batch.edge_index, batch.batch, y)
            latents.append(z.cpu().numpy())
            y_all.append(y.cpu().numpy())
            ids.extend(batch.id.cpu().numpy() if hasattr(batch, 'id') else [-1] * z.size(0))
            sesiones.extend(batch.sesion.cpu().numpy() if hasattr(batch, 'sesion') else [-1] * z.size(0))

    latents = np.concatenate(latents)
    y_all = np.concatenate(y_all)
    return latents, y_all, ids, sesiones

# Dimensionality Reduction
from sklearn.decomposition import PCA
import umap

def compute_latent_2d_projection(latents, method="tsne"):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    return reducer.fit_transform(latents)

# Plots 2D projection of latent space colored by a selected variable.
def plot_latent_projection(latents_2d, y_all, var_index, label, cmap="viridis", filter_range=None, save_name=None, save_dir="Results/Mid supervised + KL Warmup"):
    values = y_all[:, var_index]

    if filter_range:
        mask = (values > filter_range[0]) & (values < filter_range[1])
        latents_2d = latents_2d[mask]
        values = values[mask]

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=values, cmap=cmap, alpha=0.7)
    plt.colorbar(sc, label=label)
    plt.title(f"Latent Space Colored by {label}")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.grid(True)
    plt.tight_layout()

    if save_name:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, save_name)
        plt.savefig(path)
        plt.show()
        plt.close()
    else:
        plt.show()

def plot_patient_latents(latents_2d, ids, sesiones, patient_id, save_name="patient_latents.png"):
    latents = np.array(latents_2d)
    ids = np.array(ids)
    sesiones = np.array(sesiones)

    fig, ax = plt.subplots(figsize=(8, 6))

    other = latents[ids != patient_id]
    selected = latents[ids == patient_id]
    selected_sesiones = sesiones[ids == patient_id]

    ax.scatter(other[:, 0], other[:, 1], label="Otros pacientes", alpha=0.3, s=20, color="gray")

    unique_sesiones = np.unique(selected_sesiones)
    norm = plt.Normalize(vmin=min(unique_sesiones), vmax=max(unique_sesiones))
    cmap = plt.cm.plasma 

    for punto, sesion in zip(selected, selected_sesiones):
        color = cmap(norm(sesion))
        ax.scatter(punto[0], punto[1], color=color, s=80, marker="x")
        ax.text(punto[0] + 0.02, punto[1], str(sesion), fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Sesión')

    ax.set_title(f"Latent space - Paciente {patient_id}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()

def generate_model(weight_real, height_real, gender, encoder, decoder, dataset, edge_index, device, save_path="comparacion.png"):
    import trimesh
    from sklearn.decomposition import PCA

    # Normalize
    weight_mean, weight_std = 87.76, 14.75
    height_mean, height_std = 166.42, 9.18

    weight_norm = (weight_real - weight_mean) / weight_std
    height_norm = (height_real - height_mean) / height_std
    y_cond = torch.tensor([[weight_norm, height_norm, gender]], dtype=torch.float32).to(device)

    # Generate z 
    dummy_data = dataset[0].clone()
    dummy_data.x = dummy_data.x.to(device)
    dummy_data.edge_index = dummy_data.edge_index.to(device)
    dummy_data.batch = torch.zeros(dummy_data.x.size(0), dtype=torch.long).to(device)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        mu, _, z, _ = encoder(dummy_data.x, dummy_data.edge_index, dummy_data.batch, y_cond)
        pred_verts = decoder(z, y_cond).squeeze().cpu().numpy()

    candidates = [d for d in dataset if hasattr(d, "gender") and int(d.gender.item()) == gender]

    mejor_idx = -1
    mejor_dist = float("inf")
    for i, d in enumerate(candidates):
        y = d.y[0] 
        weight_diff = abs((y[0] * weight_std + weight_mean) - weight_real)
        height_diff = abs((y[1] * height_std + height_mean) - height_real)
        dist = weight_diff + 0.5 * height_diff 

        if dist < mejor_dist:
            mejor_dist = dist
            mejor_idx = i

    sample_real = candidates[mejor_idx]
    real_verts = sample_real.x.view(-1, 3).cpu().numpy()

    # Visualize
    print(f"[INFO] Comparing with nearest real sample: ID {getattr(sample_real, 'id', 'N/A')} sesion {getattr(sample_real, 'sesion', 'N/A')}")
    dists = np.linalg.norm(pred_verts - real_verts, axis=1)
    norm_dists = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)  # normalize

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Convert distances to colors using a colormap
    blue_red = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])
    colors = blue_red(norm_dists)
    colors = (colors * 255).astype(np.uint8)

    pred_pc = trimesh.PointCloud(pred_verts, colors=colors)
    pred_pc.export("predicted_colored.ply")

    real_pc = trimesh.PointCloud(real_verts, colors=[0, 255, 0, 255])
    real_pc.export("real_colored.ply")

    # Latent space projection
    all_latents, y_all, _, _ = extract_latents_and_variables(encoder, DataLoader(dataset, batch_size=16), device)
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(all_latents)

    with torch.no_grad():
        z_nuevo = encoder(dummy_data.x, dummy_data.edge_index, dummy_data.batch, y_cond)[2].cpu().numpy()
        z_nuevo_2d = pca.transform(z_nuevo)

        sample_data = torch.cat([sample_real.x], dim=0).to(device)
        sample_batch = torch.zeros(sample_data.shape[0], dtype=torch.long).to(device)
        sample_y = sample_real.y.to(device)
        z_real = encoder(sample_data, sample_real.edge_index.to(device), sample_batch, sample_y)[2].cpu().numpy()
        z_real_2d = pca.transform(z_real)

    # Values of the most similar sample
    y_real = sample_real.y[0].cpu()
    weight_real_sample = y_real[0].item() * weight_std + weight_mean
    height_real_sample = y_real[1].item() * height_std + height_mean
    gender_sample = int(round(y_real[2].item()))

    print(f"\n[INFO] Most similar sample:")
    print(f"  → weight:   {weight_real_sample:.2f} kg")
    print(f"  → height: {height_real_sample:.2f} cm")
    print(f"  → gender: {'Woman' if gender_sample == 0 else 'Men'}")

    # Graficar
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.3, label="Dataset")
    plt.scatter(z_nuevo_2d[0, 0], z_nuevo_2d[0, 1], c='red', label="Generate", marker='X', s=100)
    plt.scatter(z_real_2d[0, 0], z_real_2d[0, 1], c='green', label="Closer", marker='o', s=100)
    plt.legend()
    plt.title("PCA latent space")
    plt.savefig(save_path)
    plt.show()
