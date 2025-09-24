import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def build_edge_index_from_faces(faces):
    # faces: (F, 3) numpy or tensor
    edges = set()
    for face in faces:
        v0, v1, v2 = face
        edges.update([
            (v0, v1), (v1, v0),
            (v1, v2), (v2, v1),
            (v2, v0), (v0, v2)
        ])
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index  # shape: [2, num_edges]

class T4D_Dataset(Dataset):
    def __init__(self, x_data, y_data):
        super(T4D_Dataset, self).__init__()
        self.x = torch.Tensor(x_data)
        self.y = torch.Tensor(y_data)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
    
def load_data(df_path="df_final.csv", test_split = 0.1):
    df = pd.read_csv(df_path)
    df.drop(['fecha'], axis=1, inplace=True)

    # Create train and test using train_test_split
    train, test = train_test_split(df, test_size=test_split, random_state=42)

    train_cliente = train['cliente']
    test_cliente = test['cliente']
    train_sesion = train['sesion']
    test_sesion = test['sesion']

    train.drop('cliente', axis=1, inplace=True)
    test.drop('cliente', axis=1, inplace=True)
    train.drop('sesion', axis=1, inplace=True)
    test.drop('sesion', axis=1, inplace=True)
    
    beta_cols = [col for col in train.columns if 'beta' in col]
    X_train = train.drop(beta_cols, axis=1)
    y_train = train[beta_cols]
    X_test = test.drop(beta_cols, axis=1)
    y_test = test[beta_cols]

    # Normalize data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Normalize weight
    scaler_weight = preprocessing.StandardScaler().fit(y_train.values)
    y_train = scaler_weight.transform(y_train.values)
    y_test = scaler_weight.transform(y_test.values)

    return X_train, y_train, X_test, y_test, train_cliente, test_cliente, train_sesion, test_sesion, train, test

class RMSELoss(nn.Module):
	def __init__(self, eps=1e-6, reduction="mean"):
		super().__init__()
		self.mse = nn.MSELoss(reduction=reduction)
		self.eps = eps

	def forward(self, y_hat, y):
		return torch.sqrt(self.mse(y_hat, y)+self.eps)
     
def kl_annealing(iteration, cycles, max_iteration, max_beta=7, hold_ratio=0.1):
    if iteration >= max_iteration:
        return max_beta ** 2
    cycle_length = max_iteration // cycles
    ramp_up_length = cycle_length * (1 - hold_ratio)
    t = (iteration % cycle_length)

    if t < ramp_up_length:
        beta = (t / ramp_up_length) * max_beta
    else:
        beta = max_beta

    return beta ** 2

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def init_weights_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()
