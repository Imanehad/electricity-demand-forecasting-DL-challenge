# Electricity Demand Forecasting

Le bon fonctionnement du système électrique repose sur un équilibre constant entre l'offre et la demande d'électricité. En raison de l'impossibilité de stocker l'électricité, sa production doit être ajustée en temps réel pour correspondre à la demande.
Ce projet vise donc à développer un modèle de Deep Learning pour prédire la consommation d'électricité de 2022 en se basant sur des données historiques de consommation de 2017 à 2021 et des paramètres météorologiques. L'objectif principal est d'anticiper la demande d'électricité au niveau national, régional et pour 12 métropoles, tout en prenant en compte des facteurs météorologiques tels que la température, le vent et la pression.


## Objectifs du projet

- **Prévoir la consommation d'électricité** au niveau national, régional et pour 12 villes spécifiques.
- **Utiliser des données météorologiques** pour affiner les prévisions et rendre les modèles plus robustes.
- **Explorer des modèles de Deep Learning** pour gérer la non-stationnarité et les incertitudes liées à la consommation d'électricité.


## Structure du projet

Vous y trouverez les éléments suivants :

**Préprocessing**: 
  - Features engineering: 
-ajout de variables temporelles (weekdays, weekend, vacances scolaires par zones, saisons...)
-ajout de variables de température par régions, par métropoles et moyenne pondérée pour la France
-ajout de variable météo (vitesse et direction du vent, pression aux stations et température minimale du sol)
   - Data cleaning: gestion des NaNs

 **Training** : construction d'un modèle MLP avec CustomLoss et évaluation

**Prédict** : prédiction des consommations et export

## Fichiers

train.py : contient tout le préprocessing et l'entrainement du modèle
models.py : le modele 
predict.py : code permettant de créer le fichier de soumission
