import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import argparse
import torch
from torch_geometric.loader import DataLoader
from collections import Counter
from os import listdir
import pandas as pd
from os.path import isfile, join
from tqdm import tqdm
import numpy as np

from lib.models import Encoder, Decoder
from lib.train import train_model
from lib.dataset import load_ply_as_data, extract_info_from_files
from lib.eval import evaluate_and_visualize, extract_latents_and_variables, compute_latent_2d_projection, plot_latent_projection, plot_patient_latents, generate_model

import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EDGE_INDEX = torch.load("edge_index.pt")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_min_max(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

# Load mesh dataset and metadata, create PyG Data objects
def load_dataset(dataset_path, edge_index, patient_csv="Dataset/paciente.csv", df_path="Dataset/sesion.csv"):
    df = pd.read_csv(patient_csv)
    df_sesion = pd.read_csv(df_path)

    dni_to_id = dict(zip(df['dni'], df['id']))
    gender_map = {'M': 0, 'H': 1}
    df['gender_cod'] = df['sexo'].str.upper().str.strip().map(gender_map)
    dni_to_gender_cod = dict(zip(df['dni'], df['gender_cod']))
    dni_to_height = dict(zip(df['dni'], df['height']))

    # Extract varibles
    weights = extract_info_from_files(dataset_path, patient_csv, df_path, variable="Weight")
    fats = extract_info_from_files(dataset_path, patient_csv, df_path, variable="Total Fat")
    muscles = extract_info_from_files(dataset_path, patient_csv, df_path, variable="Total Muscle")

    # Maps
    weight_map = {(dni, sesion): float(valor) for (_, dni, sesion, valor) in weights if valor is not None}
    fat_map = {(dni, sesion): float(valor) for (_, dni, sesion, valor) in fats if valor is not None}
    muscle_map = {(dni, sesion): float(valor) for (_, dni, sesion, valor) in muscles if valor is not None}

    # Normalization parameters
    weight_mean, weight_std = 87.76, 14.75
    height_mean, height_std = 166.42, 9.18
    fat_mean, fat_std = 35.30, 7.79  
    muscle_mean, muscle_std = 61.21 , 7.79  

    ply_files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and f.endswith('.ply')]
    print("Loading mesh dataset...")
    dataset = []

    for f in tqdm(ply_files):
        path = join(dataset_path, f)
        data = load_ply_as_data(path, edge_index)

        parts = f.split("_")
        if len(parts) >= 2:
            dni = parts[0]
            sesion_str = parts[1]
            sesion = int(sesion_str[0]) if sesion_str and sesion_str[0].isdigit() else -1
            id_val = dni_to_id.get(dni, -1)
            gender = dni_to_gender_cod.get(dni, -1)
            height = dni_to_height.get(dni, 0.0)

            weight = weight_map.get((dni, sesion), 0.0)
            fat = fat_map.get((dni, sesion), 0.0)
            muscle = muscle_map.get((dni, sesion), 0.0)

            if 0.0 in (weight, fat, muscle):
                continue

            # Normalization
            weight_norm = (weight - weight_mean) / weight_std
            height_norm = (height - height_mean) / height_std
            fat_norm = (fat - fat_mean) / fat_std
            muscle_norm = (muscle - muscle_mean) / muscle_std

            data.id = torch.tensor(id_val, dtype=torch.long)
            data.sesion = torch.tensor(sesion, dtype=torch.long)
            data.gender = torch.tensor(gender, dtype=torch.long)
            data.y = torch.tensor([[weight_norm, height_norm, gender]], dtype=torch.float32)
            data.__cat_dim__ = lambda key, *_: None if key == 'y' else 0

            dataset.append(data)

    split_idx = int(len(dataset) * 0.8)
    return dataset[:split_idx], dataset[split_idx:]


def main(mode):

    set_seed(42)  # Set random seed for reproducibility

    # Initialize models
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    # Define loss weights
    loss_weights = {"kl": 1.0, "recon": 1.0, "ot": 2.0, "edge": 1.0}

    # Load dataset
    dataset_path = "Dataset/FinalDatasetPose"
    patient_csv="Dataset/paciente.csv"
    df_path="Dataset/df_final3.csv"
    train_dataset, test_dataset = load_dataset(dataset_path, EDGE_INDEX, patient_csv, df_path)

    full_dataset = train_dataset + test_dataset
    all_y = torch.cat([d.y for d in full_dataset], dim=0)

    # Print gender distribution in train set
    genders = [data.gender.item() for data in train_dataset if hasattr(data, 'gender')]
    n_women = sum(1 for g in genders if g == 0)
    n_men = sum(1 for g in genders if g == 1)
    print(f"\n[INFO] Total en train: {len(train_dataset)} samples")
    print(f"[INFO] women (gender=0): {n_women}")
    print(f"[INFO] men (gender=1): {n_men}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if mode == "train":
        print("[INFO] Training...")
        num_epochs = 50

        encoder, decoder = train_model(
            encoder, decoder,
            train_dataloader,
            num_epochs,
            device,
            loss_weights
        )

        torch.save(encoder.state_dict(), "encoder.pth")
        torch.save(decoder.state_dict(), "decoder.pth")
        print("[INFO] Training complete. Models saved to disk.")

    elif mode == "eval":
        print("[INFO] Loading models for evaluation...")

        # Charge encoder and decoder
        encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
        decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
        encoder.eval()
        decoder.eval()

        # Evaluate and visualize results
        print("[INFO] Evaluating and visualizing on test set...")
        evaluate_and_visualize(
            encoder=encoder,
            decoder=decoder,
            dataloader=test_dataloader,
            device=device,
            scene_bool=True,
            num_samples=0
        )

        latents, y_all, ids_all, sesiones = extract_latents_and_variables(encoder, train_dataloader, device)
        latents_2d = compute_latent_2d_projection(latents,method="pca")

        df_info = pd.read_csv("Dataset/df_final3.csv")

        # Exctract values
        flat_vals, muscle_vals = [], []
        for data in train_dataloader.dataset:
            id_ = data.id.item()
            sesion = data.sesion.item()
            
            row = df_info[(df_info["cliente"] == id_) & (df_info["sesion"] == sesion)]
            if not row.empty:
                flat_vals.append(row["Total Fat"].values[0])
                muscle_vals.append(row["Total Muscle"].values[0])
            else:
                flat_vals.append(0.0)  
                muscle_vals.append(0.0)

        flat_vals = np.array(flat_vals)
        muscle_vals = np.array(muscle_vals)

        flat_vals_norm = normalize_min_max(flat_vals)
        muscle_vals_norm = normalize_min_max(muscle_vals)

        print("[INFO] Latent space projection computed.")
        plot_latent_projection(latents_2d, y_all, var_index=0, label="fat", cmap="viridis",save_name="LatentSpace_fat.png")
        plot_latent_projection(latents_2d, y_all, var_index=1, label="muscle", cmap="plasma",save_name="LatentSpace_muscle.png")
        plot_latent_projection(latents_2d, y_all, var_index=2, label="gender (0=woman, 1=man)", cmap="coolwarm",save_name="LatentSpace_gender.png")
        plot_latent_projection(latents_2d, flat_vals_norm, var_index=None, label="Total Fat", cmap="cividis", save_name="LatentSpace_Flat.png")
        plot_latent_projection(latents_2d, muscle_vals_norm, var_index=None, label="Total Muscle", cmap="magma", save_name="LatentSpace_Muscle.png")
        plot_patient_latents(latents_2d, ids_all, sesiones, patient_id=84, save_name="Paciente23.png")
        generate_model(
            weight_real=113.0,
            height_real=178.0,
            gender=1,
            encoder=encoder,
            decoder=decoder,
            dataset=train_dataset + test_dataset,
            edge_index=EDGE_INDEX,
            device=device
        )


    else:
        raise ValueError("Invalid mode. Use --mode train or --mode eval")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Run mode: train or eval")
    args = parser.parse_args()

    main(args.mode)
