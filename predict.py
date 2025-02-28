import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
import holidays
import datetime
import pytz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from math import radians, sin, cos, sqrt, atan2
from geopy.geocoders import Nominatim
import time
from models.py import PytorchPerceptron, CustomLoss



"""### ***Prédiction***"""
model = PytorchPerceptron(input_shape=205, output_shape=25, dropout_rate=0.3)
model.load_state_dict(torch.load("modele.pth"))
model.eval()

test = pd.read_csv('test_final.csv')
test_df = test.copy()

scaler = StandardScaler()
X_test = scaler.transform(test_df)
X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)


pred=model(X_test)
pred_df = pd.read_csv('pred.csv')

target_col = ['France', 'Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
    'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France', 'Normandie',
    'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire', "Provence-Alpes-Côte d'Azur",
    'Île-de-France','Montpellier Méditerranée Métropole', 'Métropole Européenne de Lille',
    'Métropole Grenoble-Alpes-Métropole', "Métropole Nice Côte d'Azur",
    'Métropole Rennes Métropole', 'Métropole Rouen Normandie',
    "Métropole d'Aix-Marseille-Provence", 'Métropole de Lyon',
    'Métropole du Grand Nancy', 'Métropole du Grand Paris',
    'Nantes Métropole', 'Toulouse Métropole']

target_indices = {target: i for i, target in enumerate(target_col)}

for target in target_col:
    pred_column = f'pred_{target}'
    if pred_column in pred_df.columns:
        pred_df[pred_column] =pred.detach().numpy()[:, target_indices[target]]

pred_df.to_csv('pred.csv', index=False)