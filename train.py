# -*- coding: utf-8 -*-
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


"""# **PRÉ PROCESSING**

# *Features engineering*

## ***Train***

Conversion de la date en format datetime de pandas
"""

train = pd.read_csv('train.csv')
train['date'] = pd.to_datetime(train['date'], errors='coerce', utc=True)
train.dropna(subset=['date'], inplace=True)

"""### ***Ajout de variables temporelles***"""

train['date'] = train['date'].dt.tz_convert('Europe/Paris')
train['mois'] = train['date'].dt.month
train['jour'] = train['date'].dt.day
train['heure'] = train['date'].dt.hour
train['jour_semaine'] = train['date'].dt.weekday
train['weekend'] = np.where(train['date'].dt.weekday >= 5, 1, 0)

train.describe()

# ajout de la variable changement d'huere
def changement_heure(date):
    annee = date.year
    last_sunday_march = max(
        datetime.date(annee, 3, day) for day in range(25, 32)
        if datetime.date(annee, 3, day).weekday() == 6
    )
    last_sunday_october = max(
        datetime.date(annee, 10, day) for day in range(25, 32)
        if datetime.date(annee, 10, day).weekday() == 6
    )
    return 1 if date.date() == last_sunday_march or date.date() == last_sunday_october else 0
train['changement_heure'] = train['date'].apply(changement_heure)


# ajout des jours fériés
fr_holidays = holidays.France()
train['jour_férié'] = train['date'].apply(lambda x: 1 if x in fr_holidays else 0)


# ajout des vacances scolaires par zones
def vacances_scolaires(date, zone='A'):
    vacances = {
    2017: {
        'Toussaint': (pd.Timestamp('2017-10-21', tz='Europe/Paris'), pd.Timestamp('2017-11-06', tz='Europe/Paris')),
        'Noel': (pd.Timestamp('2017-12-23', tz='Europe/Paris'), pd.Timestamp('2018-01-08', tz='Europe/Paris')),
        'Hiver': {
            'A': (pd.Timestamp('2018-02-10', tz='Europe/Paris'), pd.Timestamp('2018-02-26', tz='Europe/Paris')),
            'B': (pd.Timestamp('2018-02-24', tz='Europe/Paris'), pd.Timestamp('2018-03-12', tz='Europe/Paris')),
            'C': (pd.Timestamp('2018-02-17', tz='Europe/Paris'), pd.Timestamp('2018-03-05', tz='Europe/Paris'))
        },
        'Printemps': {
            'A': (pd.Timestamp('2018-04-07', tz='Europe/Paris'), pd.Timestamp('2018-04-23', tz='Europe/Paris')),
            'B': (pd.Timestamp('2018-04-21', tz='Europe/Paris'), pd.Timestamp('2018-05-07', tz='Europe/Paris')),
            'C': (pd.Timestamp('2018-04-14', tz='Europe/Paris'), pd.Timestamp('2018-04-30', tz='Europe/Paris'))
        }
    },
    2018: {
        'Toussaint': (pd.Timestamp('2018-10-20', tz='Europe/Paris'), pd.Timestamp('2018-11-05', tz='Europe/Paris')),
        'Noel': (pd.Timestamp('2018-12-22', tz='Europe/Paris'), pd.Timestamp('2019-01-07', tz='Europe/Paris')),
        'Hiver': {
            'A': (pd.Timestamp('2019-02-16', tz='Europe/Paris'), pd.Timestamp('2019-03-04', tz='Europe/Paris')),
            'B': (pd.Timestamp('2019-02-09', tz='Europe/Paris'), pd.Timestamp('2019-02-25', tz='Europe/Paris')),
            'C': (pd.Timestamp('2019-02-23', tz='Europe/Paris'), pd.Timestamp('2019-03-11', tz='Europe/Paris'))
        },
        'Printemps': {
            'A': (pd.Timestamp('2019-04-13', tz='Europe/Paris'), pd.Timestamp('2019-04-29', tz='Europe/Paris')),
            'B': (pd.Timestamp('2019-04-06', tz='Europe/Paris'), pd.Timestamp('2019-04-23', tz='Europe/Paris')),
            'C': (pd.Timestamp('2019-04-20', tz='Europe/Paris'), pd.Timestamp('2019-05-06', tz='Europe/Paris'))
        }
    },
    2019: {
        'Toussaint': (pd.Timestamp('2019-10-19', tz='Europe/Paris'), pd.Timestamp('2019-11-04', tz='Europe/Paris')),
        'Noel': (pd.Timestamp('2019-12-21', tz='Europe/Paris'), pd.Timestamp('2020-01-06', tz='Europe/Paris')),
        'Hiver': {
            'A': (pd.Timestamp('2020-02-15', tz='Europe/Paris'), pd.Timestamp('2020-03-02', tz='Europe/Paris')),
            'B': (pd.Timestamp('2020-02-08', tz='Europe/Paris'), pd.Timestamp('2020-02-24', tz='Europe/Paris')),
            'C': (pd.Timestamp('2020-02-22', tz='Europe/Paris'), pd.Timestamp('2020-03-09', tz='Europe/Paris'))
        },
        'Printemps': {
            'A': (pd.Timestamp('2020-04-11', tz='Europe/Paris'), pd.Timestamp('2020-04-27', tz='Europe/Paris')),
            'B': (pd.Timestamp('2020-04-04', tz='Europe/Paris'), pd.Timestamp('2020-04-20', tz='Europe/Paris')),
            'C': (pd.Timestamp('2020-04-18', tz='Europe/Paris'), pd.Timestamp('2020-05-04', tz='Europe/Paris'))
        }
    },
    2020: {
        'Toussaint': (pd.Timestamp('2020-10-17', tz='Europe/Paris'), pd.Timestamp('2020-11-02', tz='Europe/Paris')),
        'Noel': (pd.Timestamp('2020-12-19', tz='Europe/Paris'), pd.Timestamp('2021-01-04', tz='Europe/Paris')),
        'Hiver': {
            'A': (pd.Timestamp('2021-02-06', tz='Europe/Paris'), pd.Timestamp('2021-02-22', tz='Europe/Paris')),
            'B': (pd.Timestamp('2021-02-20', tz='Europe/Paris'), pd.Timestamp('2021-03-08', tz='Europe/Paris')),
            'C': (pd.Timestamp('2021-02-13', tz='Europe/Paris'), pd.Timestamp('2021-03-01', tz='Europe/Paris'))
        },
        'Printemps': {
            'A': (pd.Timestamp('2021-04-10', tz='Europe/Paris'), pd.Timestamp('2021-04-26', tz='Europe/Paris')),
            'B': (pd.Timestamp('2021-04-24', tz='Europe/Paris'), pd.Timestamp('2021-05-10', tz='Europe/Paris')),
            'C': (pd.Timestamp('2021-04-17', tz='Europe/Paris'), pd.Timestamp('2021-05-03', tz='Europe/Paris'))
        }
    },
    2021: {
        'Toussaint': (pd.Timestamp('2021-10-23', tz='Europe/Paris'), pd.Timestamp('2021-11-08', tz='Europe/Paris')),
        'Noel': (pd.Timestamp('2021-12-18', tz='Europe/Paris'), pd.Timestamp('2022-01-03', tz='Europe/Paris')),
        'Hiver': {
            'A': (pd.Timestamp('2022-02-12', tz='Europe/Paris'), pd.Timestamp('2022-02-28', tz='Europe/Paris')),
            'B': (pd.Timestamp('2022-02-05', tz='Europe/Paris'), pd.Timestamp('2022-02-21', tz='Europe/Paris')),
            'C': (pd.Timestamp('2022-02-19', tz='Europe/Paris'), pd.Timestamp('2022-03-07', tz='Europe/Paris'))
        },
        'Printemps': {
            'A': (pd.Timestamp('2022-04-16', tz='Europe/Paris'), pd.Timestamp('2022-05-02', tz='Europe/Paris')),
            'B': (pd.Timestamp('2022-04-09', tz='Europe/Paris'), pd.Timestamp('2022-04-25', tz='Europe/Paris')),
            'C': (pd.Timestamp('2022-04-23', tz='Europe/Paris'), pd.Timestamp('2022-05-09', tz='Europe/Paris'))
        }
    },
    2022: {
        'Toussaint': (pd.Timestamp('2022-10-22', tz='Europe/Paris'), pd.Timestamp('2022-11-07', tz='Europe/Paris')),
        'Noel': (pd.Timestamp('2022-12-17', tz='Europe/Paris'), pd.Timestamp('2023-01-03', tz='Europe/Paris')),
        'Hiver': {
            'A': (pd.Timestamp('2023-02-04', tz='Europe/Paris'), pd.Timestamp('2023-02-20', tz='Europe/Paris')),
            'B': (pd.Timestamp('2023-02-11', tz='Europe/Paris'), pd.Timestamp('2023-02-27', tz='Europe/Paris')),
            'C': (pd.Timestamp('2023-02-18', tz='Europe/Paris'), pd.Timestamp('2023-03-06', tz='Europe/Paris'))
        },
        'Printemps': {
            'A': (pd.Timestamp('2023-04-08', tz='Europe/Paris'), pd.Timestamp('2023-04-24', tz='Europe/Paris')),
            'B': (pd.Timestamp('2023-04-15', tz='Europe/Paris'), pd.Timestamp('2023-05-02', tz='Europe/Paris')),
            'C': (pd.Timestamp('2023-04-22', tz='Europe/Paris'), pd.Timestamp('2023-05-09', tz='Europe/Paris'))
        }
    }
}
    year = date.year

    # on gere les dates en début d'année qui peuvent correspondre aux vacances de Noël de l'année précédente
    if year not in vacances and (year - 1) in vacances:
        vac = vacances[year - 1]
    elif year in vacances:
        vac = vacances[year]
    else:
        return 0

    # verif pour Toussaint, Noël
    for key in ['Toussaint', 'Noel']:
         start, end = vac[key]
         if start <= date <= end:
             return 1

    # verif pour Hiver et Printemps
    for period in ['Hiver', 'Printemps']:
         if period in vac and zone in vac[period]:
             start, end = vac[period][zone]
             if start <= date <= end:
                 return 1
    return 0
train['vacances_scolaires'] = train['date'].apply(vacances_scolaires)


# ajout des périodes de confinement
def confinement(date):
    if (
        pd.Timestamp('2020-03-17', tz='Europe/Paris') <= date <= pd.Timestamp('2020-05-11', tz='Europe/Paris')
        or pd.Timestamp('2020-10-30', tz='Europe/Paris') <= date <= pd.Timestamp('2020-12-15', tz='Europe/Paris')
        or pd.Timestamp('2021-04-03', tz='Europe/Paris') <= date <= pd.Timestamp('2021-05-03', tz='Europe/Paris')
    ):
        return 1
    return 0
train['confinement'] = train['date'].apply(confinement)


# ajout de la saison
def get_saison(date):
    year = date.year
    saison = {
        'hiver': [(pd.Timestamp(f'{year}-12-21', tz='Europe/Paris'), pd.Timestamp(f'{year+1}-03-20', tz='Europe/Paris'))],
        'printemps': [(pd.Timestamp(f'{year}-03-21', tz='Europe/Paris'), pd.Timestamp(f'{year}-06-20', tz='Europe/Paris'))],
        'été': [(pd.Timestamp(f'{year}-06-21', tz='Europe/Paris'), pd.Timestamp(f'{year}-09-22', tz='Europe/Paris'))],
        'automne': [(pd.Timestamp(f'{year}-09-23', tz='Europe/Paris'), pd.Timestamp(f'{year}-12-20', tz='Europe/Paris'))]
    }
    for saison, periods in saison.items():
        for start, end in periods:
            if start <= date <= end:
                return saison
    return 'hiver'  # Valeur par défaut

train['saison'] = train['date'].apply(get_saison)



"""### ***Encodage des variables temporelles***

Encodage cyclique
"""

train['mois_sin'] = np.sin(2 * np.pi * train['mois'] / 12)
train['mois_cos'] = np.cos(2 * np.pi * train['mois'] / 12)

train['jour_sin'] = np.sin(2 * np.pi * train['jour'] / 31)
train['jour_cos'] = np.cos(2 * np.pi * train['jour'] / 31)

train['jour_semaine_sin'] = np.sin(2 * np.pi * train['jour_semaine'] / 7)
train['jour_semaine_cos'] = np.cos(2 * np.pi * train['jour_semaine'] / 7)

train['heure_sin'] = np.sin(2 * np.pi * train['heure'] / 24)
train['heure_cos'] = np.cos(2 * np.pi * train['heure'] / 24)

season_mapping = {'hiver': 0, 'printemps': 1, 'été': 2, 'automne': 3}
train['saison_numeric'] = train['saison'].map(season_mapping)
train['saison_sin'] = np.sin(2 * np.pi * train['saison_numeric'] / 4)
train['saison_cos'] = np.cos(2 * np.pi * train['saison_numeric'] / 4)

"""Encodage de la variable saison"""

encoder = LabelEncoder()
data = train['saison']
train['saison'] = encoder.fit_transform(data)





"""## ***Test***

Nous procédons de la même manière et ajoutons les mêmes variables
"""

test = pd.read_csv('test.csv')
test['date'] = pd.to_datetime(test['date'], errors='coerce', utc=True)
test.dropna(subset=['date'], inplace=True)

test['date'] = test['date'].dt.tz_convert('Europe/Paris')
test['mois']    = test['date'].dt.month
test['jour']    = test['date'].dt.day
test['heure']   = test['date'].dt.hour
test['jour_semaine'] = test['date'].dt.weekday
test['weekend'] = np.where(test['date'].dt.weekday >= 5, 1, 0)
test['changement_heure'] = test['date'].apply(changement_heure)
test['jour_férié'] = test['date'].apply(lambda x: 1 if x in fr_holidays else 0)
test['vacances_scolaires'] = test['date'].apply(vacances_scolaires)
test['confinement'] = test['date'].apply(confinement)
test['saison'] = test['date'].apply(get_saison)

print(test.describe())

test['mois_sin'] = np.sin(2 * np.pi * test['mois'] / 12)
test['mois_cos'] = np.cos(2 * np.pi * test['mois'] / 12)

test['jour_sin'] = np.sin(2 * np.pi * test['jour'] / 31)
test['jour_cos'] = np.cos(2 * np.pi * test['jour'] / 31)

test['jour_semaine_sin'] = np.sin(2 * np.pi * test['jour_semaine'] / 7)
test['jour_semaine_cos'] = np.cos(2 * np.pi * test['jour_semaine'] / 7)

test['heure_sin'] = np.sin(2 * np.pi * test['heure'] / 24)
test['heure_cos'] = np.cos(2 * np.pi * test['heure'] / 24)

season_mapping = {'hiver': 0, 'printemps': 1, 'été': 2, 'automne': 3}
test['saison_numeric'] = test['saison'].map(season_mapping)
test['saison_sin'] = np.sin(2 * np.pi * test['saison_numeric'] / 4)
test['saison_cos'] = np.cos(2 * np.pi * test['saison_numeric'] / 4)
encoder = LabelEncoder()
data = test['saison']
test['saison'] = encoder.fit_transform(data)





"""## ***Météo***

Conversion de la date dans le même fuseau horaire que les fichiers train et test
"""

meteo = pd.read_parquet('meteo.parquet')
meteo['date'] = pd.to_datetime(meteo['date'], errors='coerce', utc=True)
meteo['date'] = meteo['date'].dt.tz_convert(pytz.FixedOffset(60))  # UTC+1
meteo = meteo.sort_values(by='date')


"""
### ***Création des variables températures par régions***
"""

meteo0 = meteo[['date','t','nom_reg']]

"""Nous procèdons à l'interpolation linéaire des températures sur une plage de dates pour chaque jour de 00:00 à 23:30, toutes les demies heures"""

meteo0 = meteo0.groupby(['date', 'nom_reg']).mean().reset_index()
new_dates = []

for region, group in meteo0.groupby('nom_reg'):
    all_days = pd.date_range(group['date'].min().floor('D'), group['date'].max().ceil('D'), freq='D')
    full_dates_per_region = pd.date_range(start=all_days.min(), end=all_days.max() + pd.Timedelta(hours=23, minutes=30), freq="30min")
    interpolated_temps = group.set_index('date').reindex(full_dates_per_region).interpolate(method='linear')['t'].reset_index()

    new_df = pd.DataFrame({
        'date': full_dates_per_region,
        'nom_reg': region,
        't': interpolated_temps['t']
    })
    new_dates.append(new_df)

meteo_30min = pd.concat(new_dates, ignore_index=True)
meteo0 = meteo_30min.interpolate(method='linear')
meteo0

"""Étant donné la différence de fuseau horaire initial, les deux premieres données sont manquantes:"""

meteo0.loc[0, 't'] = '271'
meteo0.loc[1, 't'] = '270.90'

"""Pivot du tableau pour obtenir une colonne par température par région et séparation de l'année 2022"""

meteo0 = meteo0.pivot(index="date", columns="nom_reg", values="t").reset_index()
meteo0 = meteo0.rename(columns=lambda x: f"temp_{x}" if x != "date" else x)
meteo0.columns.name = None

meteo_2022_0 = meteo0[meteo0['date'].dt.year == 2022]
meteo0 = meteo0[meteo0['date'].dt.year != 2022]

"""Fusion des températures par régions aux train et test"""

train= train.merge(meteo0, on="date", how="left")
train

train.columns

test= test.merge(meteo_2022_0, on="date", how="left")

test.columns



"""### ***Création des variables températures par métropoles***

"""

meteo1 = meteo[['date','t','numer_sta']]

"""####Interpolation linéaire

Nous procédons de même à l'interpolation, au pivot du tableau et à la séparation de l'année 2022
"""

meteo1 = meteo1.groupby(['date', 'numer_sta']).mean().reset_index()
new_dates = []

for station, group in meteo1.groupby('numer_sta'):
    all_days = pd.date_range(group['date'].min().floor('D'), group['date'].max().ceil('D'), freq='D')
    full_dates_per_station = pd.date_range(start=all_days.min(), end=all_days.max() + pd.Timedelta(hours=23, minutes=30), freq="30min")
    interpolated_temps = group.set_index('date').reindex(full_dates_per_station).interpolate(method='linear')['t'].reset_index()

    new_df = pd.DataFrame({
        'date': full_dates_per_station,
        'numer_sta': station,
        't': interpolated_temps['t']
    })
    new_dates.append(new_df)

meteo1_30min = pd.concat(new_dates, ignore_index=True)
meteo1 = meteo1_30min.interpolate(method='linear')
meteo1.loc[0, 't'] = '269.25'
meteo1.loc[1, 't'] = '269.25'
meteo1

meteo1 = meteo1.pivot(index="date", columns="numer_sta", values="t").reset_index()
meteo1.columns.name = None
meteo1

meteo_2022_1 = meteo1[meteo1['date'].dt.year == 2022]
meteo1 = meteo1[meteo1['date'].dt.year != 2022]

"""####Association des stations les plus proches des métropoles

Recherche des coordonnées gps des métropoles
"""

metropoles = [
    "Montpellier Méditerranée Métropole", "Métropole Européenne de Lille",
    "Métropole Grenoble-Alpes-Métropole", "Métropole Nice Côte d'Azur",
    "Métropole Rennes Métropole", "Métropole Rouen Normandie",
    "Métropole d'Aix-Marseille-Provence", "Métropole de Lyon",
    "Métropole du Grand Nancy", "Métropole du Grand Paris",
    "Nantes Métropole", "Toulouse Métropole"
]

geolocator = Nominatim(user_agent="geoapi")

coords = {"nom_metropole": [], "latitude": [], "longitude": []}
for metro in metropoles:
    try:
        location = geolocator.geocode(metro + ", France", timeout=10)
        if location:
            coords["nom_metropole"].append(metro)
            coords["latitude"].append(location.latitude)
            coords["longitude"].append(location.longitude)
            print(f"{metro} → {location.latitude}, {location.longitude}")
        else:
            print(f"Coordonnées non trouvées pour {metro}")
            coords["nom_metropole"].append(metro)
            coords["latitude"].append(None)
            coords["longitude"].append(None)
    except Exception as e:
        print(f"Erreur pour {metro}: {e}")
    time.sleep(1)  # arret de 1 sec à chaque itération pour éviter le blocage de l'API

df_metropoles = pd.DataFrame(coords)
df_metropoles.to_csv("coordonnees_metropoles.csv", index=False)

"""Calcul de distance entre les métropoles et les stations graces aux coordonnées, et association de chaque métropole à la station la plus proche."""

df_metropoles = pd.read_csv("coordonnees_metropoles.csv")
df_stations = meteo[["numer_sta", "latitude", "longitude"]]

def calcul_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

metropole_station = {}
for _, metro in df_metropoles.iterrows():
    min_distance = float("inf")
    station_proche = None
    for _, station in df_stations.iterrows():
        distance = calcul_distance(metro["latitude"], metro["longitude"], station["latitude"], station["longitude"])
        if distance < min_distance:
            min_distance = distance
            station_proche = station["numer_sta"]

    metropole_station[metro["nom_metropole"]] = station_proche
    print(f" {metro['nom_metropole']} → {station_proche} ({min_distance:.2f})")

df_assoc = pd.DataFrame(list(metropole_station.items()), columns=["Metropole", "Station_Meteo"])
df_assoc.to_csv("metropoles_stations_assoc.csv", index=False)

"""Fusion des températures des stations les plus proches des métropoles avec le train"""

station_metropole = {v: k for k, v in metropole_station.items()}

# filtre des stations associées aux métropoles
stations_associees = [station for station, metropole in station_metropole.items() if metropole in metropoles]
meteo_filtre = meteo1[['date'] + [str(station) for station in stations_associees]]
meteo_filtre['date'] = pd.to_datetime(meteo_filtre['date'])

for station in stations_associees:
    train = train.merge(meteo_filtre[['date', str(station)]],on='date', how='left')
    metropole = station_metropole.get(str(station), None)
    if metropole:
        train.rename(columns={str(station): f'temperature_{metropole}'}, inplace=True)

train.columns

"""De même pour le fichier test"""

stations_associees = [station for station, metropole in station_metropole.items() if metropole in metropoles]
meteo_filtre_2022 = meteo_2022_1[['date'] + [str(station) for station in stations_associees]]
meteo_filtre_2022['date'] = pd.to_datetime(meteo_filtre_2022['date'])

for station in stations_associees:
    test = test.merge(meteo_filtre_2022[['date', str(station)]],on='date', how='left')
    metropole = station_metropole.get(str(station), None)
    if metropole:
        test.rename(columns={str(station): f'temperature_{metropole}'}, inplace=True)

test.columns



"""### ***Création d'une variable température moyenne pondérée de la France***

Nous nous basons sur les températures moyennes des régions auxquelles nous associons un poids en fonction de la consommation d'electricité de ces dernieres
"""

region_cols = [col for col in [
    'Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
    'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France', 'Normandie',
    'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire',
    "Provence-Alpes-Côte d'Azur", 'Île-de-France'
] if col in train.columns]

temp_cols = [col for col in train.columns if col.startswith('temp_')]

region_weights = train[region_cols].mean() / train[region_cols].mean().sum()
train["Température_Pondérée"] = (train[temp_cols].values * region_weights.values).sum(axis=1)

"""De même pour le fichier test"""

temp_cols_test = [
    'temp_Auvergne-Rhône-Alpes', 'temp_Bourgogne-Franche-Comté', 'temp_Bretagne',
    'temp_Centre-Val de Loire', 'temp_Grand Est', 'temp_Hauts-de-France',
    'temp_Normandie', 'temp_Nouvelle-Aquitaine', 'temp_Occitanie',
    'temp_Pays de la Loire', "temp_Provence-Alpes-Côte d'Azur", 'temp_Île-de-France'
]

temp_cols_test = [col for col in temp_cols_test if col in test.columns]

test["Température_Pondérée"] = (test[temp_cols_test].values * region_weights.values).sum(axis=1)
test.columns



"""### ***Ajout de variables météo: vitesse et direction du vent, pression aux stations et temperature minimale du sol***"""

meteo2 = meteo[['date','numer_sta','dd', 'ff', 'pres','tminsol']]

"""Nous procédons à l'interppolation de la même manière"""

meteo2 = meteo2.groupby(['date', 'numer_sta']).mean().reset_index()

new_dates = []

for station, group in meteo2.groupby('numer_sta'):
    all_days = pd.date_range(group['date'].min().floor('D'), group['date'].max().ceil('D'), freq='D')
    full_dates_per_sta = pd.date_range(start=all_days.min(), end=all_days.max() + pd.Timedelta(hours=23, minutes=30), freq="30min")
    interpolated_temps = group.set_index('date').reindex(full_dates_per_sta).interpolate(method='linear').reset_index()

    new_df = pd.DataFrame({
        'date': full_dates_per_sta,
        'numer_sta': station,
        'dd': interpolated_temps['dd'],
        'ff': interpolated_temps['ff'],
        'pres': interpolated_temps['pres'],
        'tminsol': interpolated_temps['tminsol']

    })
    new_dates.append(new_df)

meteo_30min = pd.concat(new_dates, ignore_index=True)
meteo2 = meteo_30min.interpolate(method='linear')
meteo2

"""Nous remplissons les valeurs manquantes dues au décalage horaire par plus proche voisin"""

meteo2.loc[0, 'dd'] = '0.0'
meteo2.loc[1, 'dd'] = '0.0'
meteo2.loc[0, 'ff'] = '0.0'
meteo2.loc[1, 'ff'] = '0.0'
meteo2.loc[0, 'pres'] = '101800.0000'
meteo2.loc[1, 'pres'] = '101800.0000'
meteo2.loc[0, 'tminsol'] = '272.45'
meteo2.loc[1, 'tminsol'] = '272.45'
meteo2

"""Pivot pour obtenir une colonne par caracteristique par station"""

meteo2 = meteo2.pivot(index="date", columns="numer_sta", values=['dd', 'ff', 'pres', 'tminsol'])
meteo2.columns = [f"{col[0]}_{col[1]}" for col in meteo2.columns]
meteo2.reset_index(inplace=True)
meteo2

meteo2.columns.name = None
meteo2_2022 = meteo2[meteo2['date'].dt.year == 2022]
meteo2 = meteo2[meteo2['date'].dt.year != 2022]

"""Fusion de ces données avec les dataframes train et test"""

train = train.merge(meteo2, on="date", how="left")
test = test.merge(meteo2_2022, on="date", how="left")





"""# *Data cleaning*"""

dates_train = train['date'].copy()
train.drop(columns=['date'], inplace=True)

dates_test = test['date'].copy()
test.drop(columns=['date'], inplace=True)

nan_val = train.isnull().sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=nan_val.index, y=nan_val.values)
plt.xticks(rotation=90)
plt.title("Nombre de valeurs manquantes par colonne")
plt.xlabel("Colonnes")
plt.ylabel("Nombre de NaN")
plt.tight_layout()
plt.show()

train = train.dropna()

plt.figure(figsize=(10, 6))
sns.heatmap(train.isna(), cbar=False, cmap="viridis")
plt.title("NaN values location dans train0")
plt.show()

nan_val = test.isnull().sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=nan_val.index, y=nan_val.values)
plt.xticks(rotation=90)
plt.title("Nombre de valeurs manquantes par colonne")
plt.xlabel("Colonnes")
plt.ylabel("Nombre de NaN")
plt.tight_layout()
plt.show()

test.to_csv("test_final.csv", index=False)



"""# **ENTRAINEMENT**

### ***Construction du MLP et de la CustomLoss***"""

class PytorchPerceptron(nn.Module):
    def __init__(self, input_shape, output_shape, activation=torch.nn.ReLU(), dropout_rate=0.3):
        super().__init__()
        self.W1 = nn.Linear(input_shape, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.W2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.W3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.W4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.W5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)

        self.W6 = nn.Linear(64, output_shape)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.bn1(self.W1(x)))
        x = self.dropout1(x)
        x = self.activation(self.bn2(self.W2(x)))
        x = self.dropout2(x)
        x = self.activation(self.bn3(self.W3(x)))
        x = self.activation(self.bn4(self.W4(x)))
        x = self.activation(self.bn5(self.W5(x)))
        out = self.W6(x)
        return out

class CustomLoss(nn.Module):
    def __init__(self, france_weight, summer_weight, winter_weight, covid_weight, vacances_weight):
        super(CustomLoss, self).__init__()
        self.france_weight = france_weight
        self.summer_weight = summer_weight
        self.winter_weight = winter_weight
        self.covid_weight = covid_weight
        self.vacances_weight = vacances_weight

    def forward(self, y_pred, y_true, saison, covid, vacances):
        mse = (y_pred - y_true) ** 2
        # poids selon la saison: saison=1 correspond à l'hiver et saison=3 à l'été
        season_weight = torch.ones_like(mse)
        season_weight = torch.where(saison.unsqueeze(1)==3, self.summer_weight, season_weight)
        season_weight = torch.where(saison.unsqueeze(1)==1, self.winter_weight, season_weight)
        # poids covid et vacances
        covid_weight = torch.where(covid.unsqueeze(1)==1, self.covid_weight, torch.ones_like(mse))
        vacances_weight = torch.where(vacances.unsqueeze(1)==1, self.vacances_weight, torch.ones_like(mse))

        # application du poids de base pour la région
        total_weight = season_weight * covid_weight * vacances_weight * self.france_weight

        loss = torch.mean(total_weight * mse)
        return loss
    

"""### ***Séparation des données***
"""

def temporal_split(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    train = df[:train_size]
    val = df[train_size:]
    return train, val



"""### ***Entrainement***"""

if _name_ == "_main_":
    train_df = train.copy()
    test_df = test.copy()

    target_col = ['France', 'Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
        'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France', 'Normandie',
        'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire', "Provence-Alpes-Côte d'Azur",
        'Île-de-France','Montpellier Méditerranée Métropole', 'Métropole Européenne de Lille',
        'Métropole Grenoble-Alpes-Métropole', "Métropole Nice Côte d'Azur",
        'Métropole Rennes Métropole', 'Métropole Rouen Normandie',
        "Métropole d'Aix-Marseille-Provence", 'Métropole de Lyon',
        'Métropole du Grand Nancy', 'Métropole du Grand Paris',
        'Nantes Métropole', 'Toulouse Métropole']

    common_columns = list(set(train_df.columns) & set(test_df.columns))
    common_columns = [col for col in common_columns if col not in target_col]

    # ajout pour loss :
    saison_idx = common_columns.index("saison")
    covid_idx = common_columns.index("confinement")
    vacances_idx = common_columns.index("vacances_scolaires")

    scaler = StandardScaler()
    train_df[common_columns] = scaler.fit_transform(train_df[common_columns])
    test_df[common_columns] = scaler.transform(test_df[common_columns])

    train, val = temporal_split(train_df)

    print(f"Split des données : Train = {len(train)}, Val = {len(val)}")


    X_train, y_train = train[common_columns], train[target_col]
    X_val, y_val = val[common_columns], val[target_col]
    X_test = test_df[common_columns]

    print(f"Dimensions de X_train : {X_train.shape}, y_train : {y_train.shape}")
    print(f"Dimensions de X_val : {X_val.shape}, y_val : {y_val.shape}")
    print(f"Dimensions de X_test : {X_test.shape}")

    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)


    'Nous affectons des poids à certaines variables en fonction de leurs pertinences dans la prédiction des consommations'

    batch_size = 2500
    nb_epochs = 300
    learning_rate = 0.01

    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]

    model = PytorchPerceptron(input_shape, output_shape, dropout_rate=0.3)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    def pytorch_training_with_custom_loss(model, train_loader, X_val, y_val, nb_epochs, criterion, optimizer, saison_idx, covid_idx, vacances_idx):
        loss_history = []
        val_loss_history = []

        for epoch in range(nb_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                saison_batch = X_batch[:, saison_idx].to(torch.int64)
                covid_batch = X_batch[:, covid_idx]
                vacances_batch = X_batch[:, vacances_idx]

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch, saison_batch, covid_batch, vacances_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                saison_val = X_val[:, saison_idx].to(torch.int64)
                covid_val = X_val[:, covid_idx]
                vacances_val = X_val[:, vacances_idx]

                y_val_pred = model(X_val)
                val_loss = criterion(y_val_pred, y_val, saison_val, covid_val, vacances_val)

            loss_history.append(loss.item())
            val_loss_history.append(val_loss.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        return model, loss_history, val_loss_history



    custom_loss = CustomLoss(france_weight=1.5, summer_weight=1.5, winter_weight=2., covid_weight=0.5, vacances_weight=1.2)

    trained_model, loss_history, val_loss_history = pytorch_training_with_custom_loss(
        model,
        train_loader,
        X_val, y_val,
        nb_epochs,
        custom_loss,
        optimizer,
        saison_idx,
        covid_idx,
        vacances_idx
    )


    plt.figure(figsize=(10,5))
    plt.plot(loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Évolution de la perte (Train vs Validation)")
    plt.show()


    model.load_state_dict(torch.load("modele.pth"))



