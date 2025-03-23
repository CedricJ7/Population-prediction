"""
Module de définition du layout de l'application.
Ce module contient les fonctions pour créer la structure visuelle de l'application.
"""

import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc

# Import des données et fonctions utilitaires
from data import data, CONTINENT_COLORS, get_year_range

def create_layout(app):
    """
    Crée le layout principal de l'application.
    
    Args:
        app (Dash): L'application Dash
        
    Returns:
        html.Div: Le layout principal
    """
    # Obtention de la plage d'années dans les données
    min_year, max_year = get_year_range(data)
    
    # Calcul des chiffres clés
    key_figures = calculate_key_figures(data)
    
    # Création du layout principal
    return html.Div(
        className="app-container",
        children=[
            # En-tête compact de l'application
            create_compact_header(),
            
            # Système d'onglets
            dcc.Tabs(id='tabs', value='tab-data', className='custom-tabs', children=[
                # Onglet Données
                dcc.Tab(
                    label='Données', 
                    value='tab-data',
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                    children=[
                        # Conteneur principal pour l'onglet Données
                        html.Div(
                            className="main-container",
                            children=[
                                # Panneau latéral (filtres et chiffres clés)
                                html.Div(
                                    className="sidebar",
                                    children=[
                                        # Filtres
                                        create_filter_with_key_figures_panel(min_year, max_year, key_figures)
                                    ]
                                ),
                                
                                # Contenu principal (graphiques)
                                html.Div(
                                    className="content-area",
                                    children=[
                                        # Visualisations principales (4 graphiques sur une vue)
                                        create_primary_content_panel()
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                
                # Onglet Prédictions
                dcc.Tab(
                    label='Prédictions', 
                    value='tab-predictions',
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                    children=[
                        # Conteneur principal pour l'onglet Prédictions
                        html.Div(
                            className="main-container",
                            children=[
                                # Contenu des prédictions
                                create_predictions_panel()
                            ]
                        )
                    ]
                )
            ]),
            
            # Pied de page compact
            create_compact_footer()
        ]
    )

def calculate_key_figures(df):
    """
    Calcule les chiffres clés à afficher.
    
    Args:
        df (DataFrame): Données démographiques
        
    Returns:
        dict: Dictionnaire contenant les chiffres clés formatés
    """
    # Données pour l'année la plus récente
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    
    # Calculer l'espérance de vie moyenne mondiale
    avg_life_expectancy = latest_data['Life expectancy at birth, total (years)'].mean()
    formatted_life_expectancy = f"{avg_life_expectancy:.1f} ans"
    
    # Calculer la population mondiale totale
    total_population = latest_data['Population, total'].sum() / 1_000_000
    if total_population >= 1_000:
        formatted_population = f"{total_population/1_000:.2f} Mrd"
    else:
        formatted_population = f"{total_population:.1f} M"
    
    # Calculer le taux de croissance démographique
    prev_year = latest_year - 5  # Utiliser l'année 5 ans plus tôt pour le calcul
    prev_data = df[df['Year'] == prev_year]
    prev_population = prev_data['Population, total'].sum()
    latest_population = latest_data['Population, total'].sum()
    years_diff = latest_year - prev_year
    growth_rate = ((latest_population / prev_population) ** (1/years_diff) - 1) * 100
    formatted_growth_rate = f"{growth_rate:.2f}%"
    
    # Calculer le PIB médian
    median_gdp = latest_data['GDP per capita (current US$)'].median()
    formatted_median_gdp = f"${median_gdp:.0f}"
    
    # Calculer la fertilité moyenne
    avg_fertility = latest_data['Birth rate, crude (per 1,000 people)'].mean()
    formatted_fertility = f"{avg_fertility:.1f}‰"
    
    return {
        'life_expectancy': formatted_life_expectancy,
        'population': formatted_population,
        'growth_rate': formatted_growth_rate,
        'median_gdp': formatted_median_gdp,
        'fertility': formatted_fertility
    }

def create_compact_header():
    """
    Crée l'en-tête compact de l'application.
    
    Returns:
        html.Header: L'en-tête de l'application
    """
    return html.Header(
        className="compact-header",
        children=[
            html.Div(
                className="header-content",
                children=[
                    html.Div(
                        className="logo-container",
                        children=[
                            html.Div(className="logo-icon"),
                            html.H1(className="header-title", children="Analyse de données socio-économiques")
                        ]
                    )
                ]
            )
        ]
    )

def create_compact_footer():
    """
    Crée le pied de page compact de l'application.
    
    Returns:
        html.Footer: Le pied de page de l'application
    """
    return html.Footer(
        className="compact-footer",
        children=[
            html.Div(
                className="footer-content",
                children=[
                    html.P(className="footer-text", children="© 2023 Tableau de bord démographique")
                ]
            )
        ]
    )

def create_filter_with_key_figures_panel(min_year, max_year, key_figures):
    """
    Crée le panneau de filtres avec les chiffres clés intégrés.
    
    Args:
        min_year (int): Année minimale disponible
        max_year (int): Année maximale disponible
        key_figures (dict): Dictionnaire contenant les chiffres clés formatés
        
    Returns:
        html.Div: Le panneau de filtres avec chiffres clés
    """
    # Sélection des variables disponibles pour les axes X et Y
    variable_options = [
        {'label': 'Espérance de vie', 'value': 'Life expectancy at birth, total (years)'},
        {'label': 'PIB par habitant', 'value': 'GDP per capita (current US$)'},
        {'label': 'Population', 'value': 'Population (Millions)'},
        {'label': 'Croissance démographique', 'value': 'Population growth (annual %)'},
        {'label': 'Taux de natalité', 'value': 'Birth rate, crude (per 1,000 people)'},
        {'label': 'Taux de mortalité', 'value': 'Death rate, crude (per 1,000 people)'}
    ]
    
    return html.Div(
        className="filter-panel",
        children=[
            html.H2(
                className="panel-title",
                children=[
                    html.I(className="fas fa-filter filter-icon"),
                    "Filtres et Indicateurs"
                ]
            ),
            
            # Filtre année
            html.Div(
                className="filter-section year-filter",
                children=[
                    html.Label(
                        className="filter-label",
                        children=[
                            html.I(className="fas fa-calendar-alt filter-icon"),
                            "Plage d'années"
                        ]
                    ),
                    html.Div(
                        className="year-range-container",
                        children=[
                            html.Div(
                                id="year-range-display",
                                className="year-range-display"
                            ),
                            dcc.RangeSlider(
                                id="year-slider",
                                min=min_year,
                                max=max_year,
                                step=1,
                                value=[1960, 2023],
                                marks={
                                    year: {'label': str(year)}
                                    for year in range(min_year, max_year + 1, 10)
                                },
                                className="year-slider"
                            )
                        ]
                    )
                ]
            ),
            
            # Filtre variable X
            html.Div(
                className="filter-section",
                children=[
                    html.Label(
                        className="filter-label",
                        children=[
                            html.I(className="fas fa-chart-line filter-icon"),
                            "Variable X"
                        ]
                    ),
                    dcc.Dropdown(
                        id="x-variable-dropdown",
                        options=variable_options,
                        value="Life expectancy at birth, total (years)",
                        clearable=False,
                        className="dropdown"
                    )
                ]
            ),
            
            # Filtre variable Y
            html.Div(
                className="filter-section filter-section-last",
                children=[
                    html.Label(
                        className="filter-label",
                        children=[
                            html.I(className="fas fa-chart-bar filter-icon"),
                            "Variable Y"
                        ]
                    ),
                    dcc.Dropdown(
                        id="y-variable-dropdown",
                        options=variable_options,
                        value="GDP per capita (current US$)",
                        clearable=False,
                        className="dropdown"
                    )
                ]
            ),
            
            # Chiffres clés intégrés (sans séparateur)
            html.Div(
                className="key-figure-container",
                children=[
                    # Espérance de vie
                    html.Div(
                        className="key-figure",
                        children=[
                            html.Div(
                                className="key-figure-icon life-expectancy-icon",
                                children=[
                                    html.I(className="fas fa-heartbeat")
                                ]
                            ),
                            html.Div(
                                className="key-figure-content",
                                children=[
                                    html.Div(
                                        id="global-life-expectancy",
                                        className="key-figure-value",
                                        children=key_figures['life_expectancy']
                                    ),
                                    html.Div(
                                        className="key-figure-label",
                                        children="Espérance de vie mondiale"
                                    )
                                ]
                            )
                        ]
                    ),
                    
                    # Population mondiale
                    html.Div(
                        className="key-figure",
                        children=[
                            html.Div(
                                className="key-figure-icon population-icon",
                                children=[
                                    html.I(className="fas fa-users")
                                ]
                            ),
                            html.Div(
                                className="key-figure-content",
                                children=[
                                    html.Div(
                                        id="global-population",
                                        className="key-figure-value",
                                        children=key_figures['population']
                                    ),
                                    html.Div(
                                        className="key-figure-label",
                                        children="Population mondiale"
                                    )
                                ]
                            )
                        ]
                    ),
                    
                    # Taux de croissance
                    html.Div(
                        className="key-figure",
                        children=[
                            html.Div(
                                className="key-figure-icon growth-rate-icon",
                                children=[
                                    html.I(className="fas fa-chart-line")
                                ]
                            ),
                            html.Div(
                                className="key-figure-content",
                                children=[
                                    html.Div(
                                        id="growth-rate",
                                        className="key-figure-value",
                                        children=key_figures['growth_rate']
                                    ),
                                    html.Div(
                                        className="key-figure-label",
                                        children="Taux de croissance annuel"
                                    )
                                ]
                            )
                        ]
                    ),
                    
                    # PIB médian
                    html.Div(
                        className="key-figure",
                        children=[
                            html.Div(
                                className="key-figure-icon gdp-icon",
                                children=[
                                    html.I(className="fas fa-dollar-sign")
                                ]
                            ),
                            html.Div(
                                className="key-figure-content",
                                children=[
                                    html.Div(
                                        id="median-gdp",
                                        className="key-figure-value",
                                        children=key_figures['median_gdp']
                                    ),
                                    html.Div(
                                        className="key-figure-label",
                                        children="PIB médian par habitant"
                                    )
                                ]
                            )
                        ]
                    ),
                    
                    # Taux de fertilité
                    html.Div(
                        className="key-figure",
                        children=[
                            html.Div(
                                className="key-figure-icon fertility-icon",
                                children=[
                                    html.I(className="fas fa-baby")
                                ]
                            ),
                            html.Div(
                                className="key-figure-content",
                                children=[
                                    html.Div(
                                        id="fertility-avg",
                                        className="key-figure-value",
                                        children=key_figures['fertility']
                                    ),
                                    html.Div(
                                        className="key-figure-label",
                                        children="Taux de natalité moyen"
                                    )
                                ]
                            )
                        ]
                    ),
                    
                    # Lien vers les sources
                    html.Div(
                        className="source-link-container",
                        children=[
                            html.A(
                                className="source-link",
                                children=[
                                    html.I(className="fas fa-info-circle source-icon"),
                                    "Source: Banque Mondiale"
                                ],
                                href="https://databank.worldbank.org/source/world-development-indicators",
                                target="_blank",
                                title="Voir les données sources"
                            )
                        ]
                    )
                ]
            )
        ]
    )

def create_primary_content_panel():
    """
    Crée le panneau principal avec les 4 graphiques principaux.
    
    Returns:
        html.Div: Le panneau de contenu principal
    """
    return html.Div(
        className="primary-content",
        children=[
            # Première rangée
            html.Div(
                className="row first-row",
                children=[
                    # Carte choroplèthe (gauche)
                    html.Div(
                        className="card balanced-card",
                        children=[
                            html.Div(
                                className="card-header",
                                children=[
                                    html.I(className="fas fa-globe card-icon"),
                                    html.H3(className="card-title", children="Répartition mondiale"),
                                    html.Span(id="choropleth-title", className="choropleth-variable")
                                ]
                            ),
                            dcc.Graph(
                                id="choropleth-map",
                                className="graph compact-graph",
                                config={'displayModeBar': False}
                            )
                        ]
                    ),
                    
                    # Line plot (droite)
                    html.Div(
                        className="card balanced-card",
                        children=[
                            html.Div(
                                className="card-header",
                                children=[
                                    html.I(className="fas fa-chart-line card-icon"),
                                    html.H3(className="card-title", children="Évolution par continent")
                                ]
                            ),
                            dcc.Graph(
                                id="line-plot",
                                className="graph compact-graph",
                                config={'displayModeBar': False}
                            )
                        ]
                    )
                ]
            ),
            
            # Deuxième rangée
            html.Div(
                className="row second-row",
                children=[
                    # Violin plot (gauche)
                    html.Div(
                        className="card",
                        children=[
                            html.Div(
                                className="card-header",
                                children=[
                                    html.I(className="fas fa-chart-area card-icon"),
                                    html.H3(className="card-title", children="Distribution par continent")
                                ]
                            ),
                            dcc.Graph(
                                id="violin-plot",
                                className="graph compact-graph",
                                config={'displayModeBar': False}
                            )
                        ]
                    ),
                    
                    # Scatter plot (droite)
                    html.Div(
                        className="card",
                        children=[
                            html.Div(
                                className="card-header",
                                children=[
                                    html.I(className="fas fa-braille card-icon"),
                                    html.H3(className="card-title", children="Corrélation")
                                ]
                            ),
                            dcc.Graph(
                                id="scatter-plot",
                                className="graph compact-graph",
                                config={'displayModeBar': False}
                            ),
                            html.Div(
                                id="correlation-stats",
                                className="correlation-stats"
                            )
                        ]
                    )
                ]
            )
        ]
    )

