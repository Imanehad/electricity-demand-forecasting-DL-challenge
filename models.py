import torch
import torch.nn as nn

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