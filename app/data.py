"""
Module pour le chargement et le prétraitement des données démographiques.
Ce module contient les fonctions pour charger, nettoyer et organiser les données
pour l'application.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Couleurs pour les continents
CONTINENT_COLORS = {
    'Africa': '#FF5733',      # Rouge orangé
    'Asia': '#33FF57',        # Vert vif
    'Europe': '#3357FF',      # Bleu vif
    'North America': '#FF33A8', # Rose
    'Oceania': '#33FFF5',     # Cyan
    'South America': '#F5FF33' # Jaune vif
}

def load_data():
    """
    Charge les données depuis les fichiers source et effectue les transformations initiales.
    
    Returns:
        DataFrame: Données démographiques traitées
    """
    try:
        # On tente d'abord de charger depuis le dossier data
        data_path = Path('../data/data_countries_imputed_iterative.csv')
        if not data_path.exists():
            # Si le fichier n'existe pas à cet emplacement, essayer un autre
            data_path = Path('data/data_countries_imputed_iterative.csv')
        
        df = pd.read_csv(data_path)
        return preprocess_data(df)
    
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        # Créer un jeu de données factice en cas d'erreur
        return create_dummy_data()

def preprocess_data(df):
    """
    Prétraite les données en supprimant les valeurs manquantes et en effectuant
    les transformations nécessaires.
    
    Args:
        df (DataFrame): Données brutes
        
    Returns:
        DataFrame: Données prétraitées
    """
    # Convertir les types de données si nécessaire
    numeric_cols = [
        "Population, total", 
        "Birth rate, crude (per 1,000 people)", 
        "Death rate, crude (per 1,000 people)", 
        "Life expectancy at birth, total (years)",
        "Population growth (annual %)", 
        "GDP per capita (current US$)"
    ]
    
    # Convertir les colonnes en numérique, en ignorant les erreurs
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Suppression des lignes avec des valeurs manquantes dans les colonnes clés
    important_cols = ["Country Name", "Continent", "Year", "Population, total", "Life expectancy at birth, total (years)"]
    important_cols = [col for col in important_cols if col in df.columns]
    
    df = df.dropna(subset=important_cols)
    
    # Ajouter une colonne pour la population en millions (pour la taille des bulles)
    if "Population, total" in df.columns:
        df["Population (Millions)"] = df["Population, total"] / 1_000_000
    
    return df

def create_dummy_data():
    """
    Crée un jeu de données factice en cas d'erreur lors du chargement des données réelles.
    
    Returns:
        DataFrame: Données factices
    """
    # Liste de continents
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    
    # Liste de pays par continent (quelques exemples)
    countries = {
        'Africa': ['South Africa', 'Nigeria', 'Kenya', 'Egypt', 'Morocco'],
        'Asia': ['China', 'Japan', 'India', 'South Korea', 'Indonesia'],
        'Europe': ['France', 'Germany', 'Italy', 'United Kingdom', 'Spain'],
        'North America': ['United States', 'Canada', 'Mexico', 'Cuba', 'Panama'],
        'Oceania': ['Australia', 'New Zealand', 'Fiji', 'Papua New Guinea', 'Solomon Islands'],
        'South America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru']
    }
    
    # Années
    years = list(range(1990, 2021))
    
    # Création du DataFrame
    data = []
    
    for continent in continents:
        for country in countries[continent]:
            for year in years:
                # Valeurs aléatoires pour les indicateurs
                life_exp = np.random.normal(65 + (0.2 * (year - 1990)), 5)
                if continent == 'Europe' or continent == 'North America':
                    life_exp += 10
                elif continent == 'Africa':
                    life_exp -= 5
                
                gdp = np.random.lognormal(mean=8 + (0.1 * (year - 1990)), sigma=1)
                if continent == 'Europe' or continent == 'North America':
                    gdp *= 3
                elif continent == 'Africa':
                    gdp /= 2
                
                population = np.random.lognormal(mean=15, sigma=1)
                if continent == 'Asia':
                    population *= 5
                elif continent == 'Oceania':
                    population /= 5
                
                birth_rate = np.random.normal(20 - (0.2 * (year - 1990)), 3)
                if continent == 'Africa':
                    birth_rate += 10
                elif continent == 'Europe':
                    birth_rate -= 8
                
                death_rate = np.random.normal(10 - (0.05 * (year - 1990)), 2)
                
                data.append({
                    'Country Name': country,
                    'Country Code': country[:3].upper(),
                    'Continent': continent,
                    'Year': year,
                    'Life expectancy at birth, total (years)': max(30, min(85, life_exp)),
                    'GDP per capita (current US$)': gdp,
                    'Population, total': population,
                    'Population (Millions)': population / 1_000_000,
                    'Birth rate, crude (per 1,000 people)': max(5, min(50, birth_rate)),
                    'Death rate, crude (per 1,000 people)': max(3, min(20, death_rate)),
                    'Population growth (annual %)': (birth_rate - death_rate) / 10
                })
    
    return pd.DataFrame(data)

def get_year_range(df):
    """
    Obtient la plage d'années disponibles dans les données.
    
    Args:
        df (DataFrame): Données démographiques
        
    Returns:
        tuple: (année_min, année_max)
    """
    return df['Year'].min(), df['Year'].max()

def filter_data_by_year_range(df, start_year, end_year):
    """
    Filtre les données par plage d'années.
    
    Args:
        df (DataFrame): Données à filtrer
        start_year (int): Année de début
        end_year (int): Année de fin
        
    Returns:
        DataFrame: Données filtrées par années
    """
    return df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

def get_correlation_stats(df, x_col, y_col):
    """
    Calcule les statistiques de corrélation entre deux colonnes.
    
    Args:
        df (DataFrame): Données
        x_col (str): Nom de la colonne X
        y_col (str): Nom de la colonne Y
        
    Returns:
        dict: Statistiques (coefficient de corrélation et pente)
    """
    # Valeurs par défaut
    default_stats = {
        'correlation': 0,
        'slope': 0,
        'intercept': 0
    }
    
    # Vérifier si le dataframe est vide ou si les colonnes n'existent pas
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print(f"Erreur: dataframe vide ou colonnes {x_col} / {y_col} manquantes")
        return default_stats
    
    # Supprimer les valeurs infinies
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Suppression des valeurs NaN pour le calcul
    data = df[[x_col, y_col]].dropna()
    
    # Vérifier si le dataframe est vide après avoir supprimé les valeurs NaN
    if data.empty or len(data) < 2:
        print("Données insuffisantes pour le calcul de corrélation après suppression des NaN")
        return default_stats
    
    try:
        # S'assurer que les données sont numériques
        data[x_col] = pd.to_numeric(data[x_col], errors='coerce')
        data[y_col] = pd.to_numeric(data[y_col], errors='coerce')
        
        # Suppression des nouvelles valeurs NaN après conversion
        data = data.dropna()
        
        # Vérifier à nouveau si suffisamment de données
        if data.empty or len(data) < 2:
            print("Données insuffisantes après conversion numérique")
            return default_stats
        
        # Coefficient de corrélation
        correlation = data.corr().iloc[0, 1]
        
        # S'assurer que correlation est un nombre valide
        if pd.isna(correlation):
            print("La corrélation calculée est NaN")
            return default_stats
        
        # Arrondir pour éviter les erreurs de virgule flottante
        correlation = round(correlation, 6)
        
        # Calcul de la pente (régression linéaire simple)
        # Convertir en array numpy 1D
        x = data[x_col].values.flatten()
        y = data[y_col].values.flatten()
        
        # Vérifier encore une fois qu'il n'y a pas de NaN
        valid_indices = ~(np.isnan(x) | np.isnan(y))
        if not np.any(valid_indices) or np.sum(valid_indices) < 2:
            print("Données insuffisantes après filtrage des NaN")
            return default_stats
        
        x = x[valid_indices]
        y = y[valid_indices]
        
        # Vérifier que x a des valeurs différentes (variance non nulle)
        if np.var(x) <= 0:
            print("La variance de x est nulle ou négative")
            return default_stats
        
        # Calcul de la pente et de l'intercept
        slope, intercept = np.polyfit(x, y, 1)
        
        # Arrondir les valeurs pour éviter les problèmes de précision
        slope = round(slope, 6)
        intercept = round(intercept, 6)
        
        return {
            'correlation': correlation,
            'slope': slope,
            'intercept': intercept
        }
    except Exception as e:
        print(f"Erreur lors du calcul des statistiques de corrélation: {e}")
        return default_stats

# Chargement des données au moment de l'importation du module
data = load_data() 