"""
Ce script permet de prévoir la population mondiale (en milliards) après 2023, à l'aide d'un modèle de croissance logistique.
Les prévisions sont affichées avec un style Seaborn amélioré :
- La courbe des données historiques est tracée avec un trait vert.
- La courbe de prévision (régression logistique) est tracée avec un trait bleu.
- Un trait pointillé gris indique la capacité maximale estimée à 11 milliards.
Le graphique est enregistré au format PNG.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Application du thème Seaborn pour un rendu graphique amélioré
sns.set_theme(style="whitegrid", palette="deep")

# ------------------------------
# Chargement et agrégation des données
# ------------------------------
data = pd.read_csv("data/data_countries_imputed.csv")
# Agrégation de la population totale par année et conversion en milliards
annual_data = data.groupby("Year")["Population, total"].sum().reset_index()
annual_data["Population, total"] = annual_data["Population, total"] / 1e9

# Extraction des données historiques (on suppose que les données vont jusqu'en 2023)
years_hist = annual_data["Year"].values
pop_hist   = annual_data["Population, total"].values

# Définition de l'horizon de prévision : de 2024 à 2100
forecast_years = np.arange(2024, 2101)

# ------------------------------
# Modèle: Régression logistique
# ------------------------------
# Normalisation des années pour améliorer la stabilité numérique
year_min = years_hist.min()
years_hist_norm = years_hist - year_min
forecast_years_norm = forecast_years - year_min

def logistic_model(x, L, k, x0):
    """
    Modèle logistique :
      L  : capacité maximale (asymptote)
      k  : taux de croissance
      x0 : point médian de la courbe
    """
    return L / (1 + np.exp(-k * (x - x0)))

# Estimation initiale des paramètres basée sur des connaissances démographiques
p0 = [11, 0.05, 70]  # [capacité maximale, taux, point médian]
params, _ = curve_fit(logistic_model, years_hist_norm, pop_hist, p0=p0, maxfev=10000)
L, k, x0 = params

forecast_logistic = logistic_model(forecast_years_norm, L, k, x0)

# ------------------------------
# Affichage du graphique comparatif
# ------------------------------
plt.figure(figsize=(14, 8))
# Courbe des données historiques en vert
plt.plot(years_hist, pop_hist, color="green", linestyle="-", linewidth=2, markersize=6, label="Données historiques")
# Courbe de prévision en bleu
plt.plot(forecast_years, forecast_logistic, color="blue", linestyle="-", linewidth=2, label="Régression logistique")

plt.xlabel("Année", fontsize=14)
plt.ylabel("Population (milliards)", fontsize=14)
plt.title("Prévision de la population mondiale après 2023", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("forecast_comparison.png")
plt.show()
