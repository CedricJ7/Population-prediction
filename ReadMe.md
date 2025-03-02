# Dashboard de Prédiction de la Population Mondiale

## Description
Ce dashboard interactif permet de visualiser et d'analyser l'évolution de la population mondiale et d'autres indicateurs démographiques de 1980 à 2023, avec des prédictions jusqu'en 2050 utilisant différents modèles d'apprentissage automatique.

## Structure du projet
Le projet est organisé en trois parties principales :

```
├── app.py           # Fichier principal contenant la logique de l'application
├── run.py           # Script pour lancer l'application
├── assets/          # Dossier contenant les ressources statiques
│   └── style.css    # Feuille de style CSS
├── templates/       # Dossier contenant les templates HTML
│   └── index.html   # Structure HTML de base
└── data.csv         # Données des indicateurs de développement mondial
```

## Configuration requise
- Python 3.8 ou supérieur
- Les bibliothèques Python suivantes :
  - dash
  - plotly
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - statsmodels

## Installation
1. Clonez ce dépôt sur votre machine locale.
2. Assurez-vous que Python 3.8+ est installé.
3. Installez les dépendances :
   ```
   pip install dash plotly pandas numpy scikit-learn xgboost statsmodels
   ```

## Utilisation
1. Placez votre fichier `data.csv` à la racine du projet.
2. Exécutez le script de lancement :
   ```
   python run.py
   ```
3. Un navigateur s'ouvrira automatiquement à l'adresse http://127.0.0.1:8050/

## Fonctionnalités
- **Analyse descriptive** : Visualisez divers indicateurs démographiques et économiques mondiaux.
  - Carte mondiale interactive
  - Graphiques d'évolution par pays
  - Tableaux de données résumées
  - Comparaisons régionales
  
- **Prédictions** : Explorez les prédictions de population mondiale jusqu'en 2050.
  - Quatre modèles de prédiction (Régression Linéaire, Random Forest, XGBoost, ARIMA)
  - Visualisation comparative des prédictions
  - Statistiques détaillées

## Personnalisation
- Vous pouvez modifier le thème de couleurs dans le fichier `assets/style.css`.
- Les pays et indicateurs affichés peuvent être ajustés dans le fichier `app.py`.

## Format des données
Le fichier `data.csv` doit contenir les colonnes suivantes :
- Country Name : Nom du pays
- Country Code : Code ISO du pays
- Series Name : Nom de l'indicateur
- Series Code : Code de l'indicateur
- [Année] [YRAnnée] : Colonnes pour chaque année (ex: 1980 [YR1980])

## Licence
Ce projet est distribué sous licence MIT.