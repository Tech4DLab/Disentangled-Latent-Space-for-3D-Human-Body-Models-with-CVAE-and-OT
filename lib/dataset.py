from torch_geometric.data import Data
import trimesh
import pandas as pd
from os import listdir
from os.path import isfile, join
import torch


def load_ply_as_data(ply_path, edge_index):
    mesh = trimesh.load(ply_path, process=False)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)  # [V, 3]
    y_center = torch.mean(vertices[:, 1])
    vertices[:, 1] -= y_center  # Center y-coordinates around 0
    return Data(x=vertices, edge_index=edge_index)

def match_dni_list_to_client_ids(dni_list, patient_csv="Dataset/paciente.csv"):
    df = pd.read_csv(patient_csv)
    dni_to_id = dict(zip(df['dni'], df['id']))
    return [dni_to_id.get(dni, None) for dni in dni_list]

def extract_info_from_files(dataset_path, patient_csv="Dataset/paciente.csv", df_final2_path="Dataset/sesion.csv", variable="Weight"):

    df_weights = pd.read_csv(df_final2_path)
    
    dni_sesion_list = []

    for filename in listdir(dataset_path):
        if filename.endswith(".ply"):
            parts = filename.split("_")
            if len(parts) >= 2:
                dni = parts[0]
                session_code = parts[1]
                if session_code and session_code[0].isdigit():
                    sesion = int(session_code[0])
                    dni_sesion_list.append((dni, sesion))
    
    dni_list = [dni for dni, _ in dni_sesion_list]
    id_list = match_dni_list_to_client_ids(dni_list, patient_csv)

    results = []

    for (dni, sesion), id_val in zip(dni_sesion_list, id_list):
        if id_val is not None:
            row = df_weights[(df_weights['cliente'] == id_val) & (df_weights['sesion'] == sesion)]
            if not row.empty and variable in row.columns:
                value = row.iloc[0][variable]
            else:
                value = None
        else:
            value = None
        results.append((id_val, dni, sesion, value))
    
    return results


