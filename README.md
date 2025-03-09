# Prédiction de la Population Mondiale jusqu'en 2100

Ce projet implémente différents modèles de prédiction pour estimer l'évolution de la population mondiale jusqu'en 2100 en utilisant des données historiques de 1960 à 2023 provenant de la Banque Mondiale.

## Modèles implémentés

Le projet utilise deux approches différentes pour modéliser et prédire la croissance démographique mondiale :

1. **Modèle ARIMA** : Utilise les techniques de séries temporelles pour capturer les dépendances temporelles dans les données.
2. **Croissance logistique** : Intègre le concept de capacité maximale et de ralentissement progressif de la croissance.

## Installation

Pour installer les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

## Utilisation

Pour exécuter l'analyse complète et générer les visualisations :

```bash
python population_forecast_2100.py
```

Cela générera trois fichiers d'image montrant :
- Les prédictions des trois modèles jusqu'en 2100
- Le modèle logistique avec intervalles de confiance croissants
- L'analyse des facteurs démographiques influençant les prévisions à long terme

## Structure des données

Le projet utilise les données de la Banque Mondiale stockées dans le dossier `data/` :
- `data_world.csv` : Données démographiques mondiales de 1960 à 2023
- Autres fichiers de données par pays et continents

## Résultats attendus

Les résultats de l'analyse suggèrent que :
- La population mondiale devrait continuer à croître jusqu'au milieu du siècle, puis se stabiliser
- Le modèle logistique suggère un plateau d'environ 10-11 milliards d'habitants vers la fin du siècle
- La baisse des taux de natalité et l'augmentation de l'espérance de vie sont des facteurs clés dans ces prévisions
- Les prévisions après 2050 présentent une incertitude croissante

## Fonctionnalités spéciales

- **Intervalles de confiance adaptatifs** : L'incertitude augmente avec l'horizon de prédiction
- **Points de repère** : Visualisation facilitée avec des marqueurs pour 2050 et 2100
- **Analyse comparative** : Projection des valeurs à court, moyen et long terme (2025, 2050, 2075, 2100)

## Limitations

Les prédictions sont sujettes à des incertitudes croissantes, particulièrement après 2050, liées à :
- Les politiques démographiques futures des différentes nations
- Les avancées technologiques imprévisibles (médecine, intelligence artificielle)
- Les changements environnementaux et leurs impacts sur les populations
- Les mouvements migratoires et les changements sociétaux à long terme
- La limitation potentielle des ressources naturelles

## Source des données

Les données utilisées proviennent de la Banque Mondiale (World Development Indicators) :
https://databank.worldbank.org/source/world-development-indicators

## Licence

[Spécifiez la licence ici]
