import dash
from dash import dcc, html, dash_table, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import webbrowser
import threading
import time
import warnings
import os
from flask_caching import Cache

warnings.filterwarnings('ignore')

# Initialize app first (required for cache)
app = dash.Dash(__name__, suppress_callback_exceptions=True,
               meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
               assets_folder='assets')
server = app.server

# Initialize cache before any @cache.memoize decorators
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Définir le thème bleu pour tout le dashboard
blue_theme = {
    'primary': '#1E88E5',       # Bleu principal
    'secondary': '#0D47A1',     # Bleu foncé
    'accent': '#64B5F6',        # Bleu clair
    'background': '#F5F7FA',    # Fond très légèrement bleuté
    'text': '#263238',          # Texte principal - presque noir
    'text_light': '#546E7A',    # Texte secondaire
    'grid': '#E1E5EA',          # Grille des graphiques
    'success': '#4CAF50',       # Vert pour valeurs positives
    'warning': '#FFC107',       # Jaune pour valeurs moyennes
    'danger': '#F44336',        # Rouge pour valeurs négatives
}

# Fonction pour charger et préparer les données
@cache.memoize(timeout=600)
def load_data():
    # Chargement des données avec l'encodage correct
    data = pd.read_csv("data.csv", encoding='cp1252')
    
    # Transformation en format long (une ligne par pays, année et indicateur)
    data_long = pd.melt(
        data,
        id_vars=["Country Name", "Country Code", "Series Name", "Series Code"],
        var_name="Year",
        value_name="Value"
    )
    
    # Extraction de l'année (juste le nombre, sans le format [YR...])
    data_long["Year"] = data_long["Year"].str.extract(r"(\d{4})").astype(int)
    
    # Pivoter les données pour avoir les indicateurs en colonnes
    data_wide = data_long.pivot_table(
        index=["Country Name", "Country Code", "Year"],
        columns="Series Name",
        values="Value",
        aggfunc='first'  # En cas de doublons, prend la première valeur
    ).reset_index()
    
    # Nettoyer les noms de colonnes
    data_wide.columns = [col.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("%", "pct").replace("$", "usd") if isinstance(col, str) else col for col in data_wide.columns]
    
    # Conversion des valeurs en numérique
    for col in data_wide.columns:
        if col not in ["country_name", "country_code", "year"]:
            data_wide[col] = pd.to_numeric(data_wide[col], errors='coerce')
    
    return data_wide

# Fonction pour extraire les données de population mondiale
@cache.memoize(timeout=600)
def get_world_population_data(data_wide):
    # Extraction des données mondiales
    world_data = data_wide[data_wide["country_code"] == "WLD"].copy()
    
    # Extraction des données de croissance et simulation de la population
    world_growth_data = world_data[["year", "population_growth_annual_pct"]].dropna().sort_values(by="year")
    
    # Population mondiale en 2023: environ 8 milliards
    BASE_POPULATION_2023 = 8000000000
    
    # Création d'un DataFrame pour la population simulée
    population_data = world_growth_data.copy()
    population_data = population_data.rename(columns={"population_growth_annual_pct": "growth_rate"})
    
    # Trouver l'année la plus récente
    if 2023 in population_data["year"].values:
        idx_2023 = population_data[population_data["year"] == 2023].index[0]
        last_year = 2023
    else:
        last_year = population_data["year"].max()
        idx_2023 = population_data[population_data["year"] == last_year].index[0]
    
    # Initialiser les valeurs de population
    population_values = [0] * len(population_data)
    population_values[idx_2023 - population_data.index[0]] = BASE_POPULATION_2023
    
    # Calculer la population des années précédentes
    for i in range(idx_2023 - population_data.index[0] - 1, -1, -1):
        growth_rate = population_data.iloc[i+1]["growth_rate"] / 100
        population_values[i] = population_values[i+1] / (1 + growth_rate)
    
    # Si l'année de référence n'est pas la dernière, calculer les années suivantes
    for i in range(idx_2023 - population_data.index[0] + 1, len(population_data)):
        growth_rate = population_data.iloc[i]["growth_rate"] / 100
        population_values[i] = population_values[i-1] * (1 + growth_rate)
    
    # Ajouter la population simulée au DataFrame
    population_data["population"] = population_values
    population_data["population_billions"] = population_data["population"] / 1000000000
    
    return population_data

# Fonction pour créer les prédictions
@cache.memoize(timeout=600)
def predict_population(population_data):
    # Préparation des données pour la modélisation
    X = population_data[["year"]].values
    y = population_data["population"].values
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Préparation des années futures pour les prédictions
    future_years = np.arange(population_data["year"].max() + 1, 2051).reshape(-1, 1)
    
    # 1. Régression linéaire
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    pred_lr_future = lr_model.predict(future_years)
    
    # 2. Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    pred_rf_future = rf_model.predict(future_years)
    
    # 3. XGBoost
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    xgb_model.fit(X_train, y_train)
    pred_xgb_future = xgb_model.predict(future_years)
    
    # 4. ARIMA
    try:
        arima_model = ARIMA(population_data["population"], order=(1, 1, 1))
        arima_fit = arima_model.fit()
        forecast = arima_fit.forecast(steps=len(future_years))
    except:
        # Modèle ARIMA simplifié en cas d'erreur
        arima_model = ARIMA(population_data["population"], order=(1, 1, 0))
        arima_fit = arima_model.fit()
        forecast = arima_fit.forecast(steps=len(future_years))
    
    # Création d'un DataFrame pour les prédictions
    predictions_df = pd.DataFrame({
        "year": future_years.flatten(),
        "linear_regression": pred_lr_future / 1000000000,  # en milliards
        "random_forest": pred_rf_future / 1000000000,
        "xgboost": pred_xgb_future / 1000000000,
        "arima": forecast.values / 1000000000
    })
    
    return predictions_df, population_data

# Chargement des données
df = load_data()
population_data = get_world_population_data(df)
predictions_df, population_data = predict_population(population_data)

# Sélection de pays représentatifs pour les graphiques
major_countries = ["USA", "CHN", "IND", "BRA", "RUS", "DEU", "GBR", "FRA", "JPN", "NGA"]
regions = ["WLD", "EAS", "ECS", "LCN", "MEA", "NAC", "SAS", "SSF"]  # Régions du monde

# Définition de la mise en page de l'application
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1('Prédiction de la Population Mondiale jusqu\'en 2050'),
            html.Div([
                html.Span('Analyse et Visualisation des Tendances Démographiques', 
                         className='subtitle')
            ])
        ], className='header-content')
    ], className='header'),
    
    html.Div([
        html.Div([
            # Sidebar (panneau de contrôle)
            html.Div([
                html.Div([
                    html.Div([
                        html.H2('Sélection des variables', className='control-panel-title'),
                        
                        dcc.Tabs(
                            id='tabs', 
                            value='tab-1', 
                            className='tabs-container',
                            children=[
                                dcc.Tab(label='Analyse Descriptive', value='tab-1', className='tab', selected_className='tab--selected'),
                                dcc.Tab(label='Prédictions', value='tab-2', className='tab', selected_className='tab--selected'),
                            ],
                            style={'margin-bottom': '20px'}
                        ),
                        
                        html.Div(id='sidebar-content')
                    ], className='control-panel')
                ])
            ], className='sidebar'),
            
            # Contenu principal (graphiques et tableaux)
            html.Div([
                html.Div(id='main-content')
            ], className='main-content')
        ])
    ], className='app-container'),
    
    # Ajout d'un footer
    html.Footer([
        html.Div([
            html.Div('© 2023 Outil de Prédiction Démographique', className='copyright'),
            html.Div([
                html.A('À propos', href='#', className='footer-link'),
                html.A('Méthodologie', href='#', className='footer-link'),
                html.A('Sources de données', href='#', className='footer-link')
            ], className='footer-links')
        ], className='footer-content')
    ], className='footer')
])

# Callback pour mettre à jour le contenu de la sidebar en fonction de l'onglet sélectionné
@app.callback(
    Output('sidebar-content', 'children'),
    Input('tabs', 'value')
)
def update_sidebar(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Div([
                html.Label('Variable pour la carte:'),
                dcc.Dropdown(
                    id='map-variable',
                    options=[
                        {'label': 'Population Growth (%)', 'value': 'population_growth_annual_pct'},
                        {'label': 'GDP per Capita', 'value': 'gdp_per_capita_current_usd'},
                        {'label': 'Life Expectancy', 'value': 'life_expectancy_at_birth_total_years'}
                    ],
                    value='population_growth_annual_pct',
                    clearable=False
                )
            ], className='input-group'),
            
            html.Div([
                html.Label('Année:'),
                dcc.Slider(
                    id='year-slider',
                    min=df['year'].min(),
                    max=df['year'].max(),
                    value=2020,
                    marks={str(year): str(year) for year in range(df['year'].min(), df['year'].max()+1, 10)},
                    step=1,
                    tooltip={'placement': 'bottom', 'always_visible': True}
                )
            ], className='input-group'),
            
            html.Div([
                html.Label('Variable pour les graphiques d\'évolution:'),
                dcc.Dropdown(
                    id='trend-variable',
                    options=[
                        {'label': 'Population Growth (%)', 'value': 'population_growth_annual_pct'},
                        {'label': 'GDP per Capita', 'value': 'gdp_per_capita_current_usd'},
                        {'label': 'Life Expectancy', 'value': 'life_expectancy_at_birth_total_years'}
                    ],
                    value='population_growth_annual_pct',
                    clearable=False
                )
            ], className='input-group'),
            
            html.Div([
                html.H3('Légende', className='control-panel-title', style={'margin-top': '30px'}),
                html.P([
                    "Ce dashboard présente les tendances démographiques mondiales de 1980 à 2023, avec des prédictions jusqu'en 2050."
                ], style={'fontSize': '13px', 'color': blue_theme['text_light']}),
                html.P([
                    "Utilisez les contrôles ci-dessus pour explorer différentes variables et périodes."
                ], style={'fontSize': '13px', 'color': blue_theme['text_light']}),
            ])
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.H3('Informations sur la prédiction', className='control-panel-title'),
                html.P([
                    "Ces prédictions sont basées sur l'analyse de la croissance démographique mondiale entre 1980 et 2023."
                ], style={'fontSize': '13px', 'color': blue_theme['text_light']}),
                html.P([
                    "Quatre modèles différents sont utilisés pour obtenir une fourchette de prédictions fiable:"
                ], style={'fontSize': '13px', 'color': blue_theme['text_light']}),
                html.Ul([
                    html.Li("Régression Linéaire", style={'color': '#E41A1C'}),
                    html.Li("Random Forest", style={'color': '#4DAF4A'}),
                    html.Li("XGBoost", style={'color': '#984EA3'}),
                    html.Li("ARIMA", style={'color': '#FF7F00'}),
                ], style={'fontSize': '13px', 'paddingLeft': '20px'}),
                html.P([
                    "Le modèle ARIMA est généralement le plus adapté pour les séries temporelles démographiques."
                ], style={'fontSize': '13px', 'color': blue_theme['text_light']}),
            ])
        ])

# Callback pour mettre à jour le contenu principal en fonction de l'onglet sélectionné
@app.callback(
    Output('main-content', 'children'),
    [Input('tabs', 'value'),
     Input('map-variable', 'value'),
     Input('year-slider', 'value'),
     Input('trend-variable', 'value')]
)
def update_main_content(tab, map_var, year, trend_var):
    selected_var = map_var if map_var is not None else 'population_growth_annual_pct'
    selected_year = year if year is not None else 2020
    selected_trend = trend_var if trend_var is not None else 'population_growth_annual_pct'
    
    if tab == 'tab-1':
        return html.Div([
            html.Div([
                # Grid container pour les 4 graphiques en 2x2
                html.Div([
                    # Carte (en haut à gauche)
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H3('Carte mondiale: ' + selected_var.replace('_', ' ').title(), 
                                      style={'margin-top': '0', 'font-size': '16px'})
                            ], className='card-header'),
                            
                            html.Div([
                                dcc.Graph(
                                    id='map-graph',
                                    config={'displayModeBar': True, 'displaylogo': False, 'responsive': True},
                                    style={'height': '100%'}
                                )
                            ], className='card-body')
                        ], className='card')
                    ], className='grid-item'),
                    
                    # Graphique d'évolution par pays (en haut à droite)
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H3('Évolution par pays: ' + selected_trend.replace('_', ' ').title(), 
                                      style={'margin-top': '0', 'font-size': '16px'})
                            ], className='card-header'),
                            
                            html.Div([
                                dcc.Graph(
                                    id='trend-graph',
                                    config={'displayModeBar': True, 'displaylogo': False, 'responsive': True},
                                    style={'height': '100%'}
                                )
                            ], className='card-body')
                        ], className='card')
                    ], className='grid-item'),
                    
                    # Tableau résumé (en bas à gauche)
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H3('Résumé des grands pays: ' + selected_var.replace('_', ' ').title(), 
                                      style={'margin-top': '0', 'font-size': '16px'})
                            ], className='card-header'),
                            
                            html.Div([
                                dash_table.DataTable(
                                    id='summary-table',
                                    style_table={'width': '100%', 'height': '100%', 'overflowY': 'auto'},
                                    style_cell={
                                        'font-family': '"Segoe UI", Arial, sans-serif',
                                        'padding': '10px 15px',
                                        'textAlign': 'center'
                                    },
                                    style_header={
                                        'backgroundColor': blue_theme['accent'],
                                        'color': 'white',
                                        'fontWeight': 'bold',
                                        'textAlign': 'center'
                                    },
                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': '#f8fafd'
                                        }
                                    ],
                                    page_size=10
                                )
                            ], className='card-body table-container')
                        ], className='card')
                    ], className='grid-item'),
                    
                    # Comparaison régionale (en bas à droite)
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H3('Comparaison régionale: ' + selected_trend.replace('_', ' ').title(), 
                                      style={'margin-top': '0', 'font-size': '16px'})
                            ], className='card-header'),
                            
                            html.Div([
                                dcc.Graph(
                                    id='region-comparison',
                                    config={'displayModeBar': True, 'displaylogo': False, 'responsive': True},
                                    style={'height': '100%'}
                                )
                            ], className='card-body')
                        ], className='card')
                    ], className='grid-item')
                ], className='grid-container')
            ])
        ])
    elif tab == 'tab-2':
        # Calculer les statistiques pour les widgets
        last_year = population_data["year"].max()
        last_pop = population_data[population_data["year"] == last_year]["population_billions"].values[0]
        
        pred_2050 = predictions_df[predictions_df["year"] == 2050]
        mean_pred_2050 = pred_2050[["linear_regression", "random_forest", "xgboost", "arima"]].mean(axis=1).values[0]
        growth_2050 = ((mean_pred_2050 / last_pop) - 1) * 100
        
        return html.Div([
            # Statistiques résumées
            html.Div([
                html.Div([
                    html.H3('Population actuelle (2023)'),
                    html.P(f'{last_pop:.2f} milliards')
                ], className='stat-card'),
                
                html.Div([
                    html.H3('Population prévue (2050)'),
                    html.P(f'{mean_pred_2050:.2f} milliards')
                ], className='stat-card'),
                
                html.Div([
                    html.H3('Croissance prévue'),
                    html.P(f'+{growth_2050:.1f}%')
                ], className='stat-card'),
                
                html.Div([
                    html.H3('Modèle le plus optimiste'),
                    html.P(f'{pred_2050[["linear_regression", "random_forest", "xgboost", "arima"]].max(axis=1).values[0]:.2f} milliards')
                ], className='stat-card')
            ], className='summary-stats'),
            
            # Graphique de prédiction et tableau
            html.Div([
                # Graphique de prédiction
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3('Prédictions de la Population Mondiale jusqu\'en 2050', 
                                  style={'margin-top': '0', 'font-size': '16px'})
                        ], className='card-header'),
                        
                        html.Div([
                            dcc.Graph(
                                id='prediction-graph',
                                config={
                                    'displayModeBar': True, 
                                    'displaylogo': False, 
                                    'responsive': True,
                                    'toImageButtonOptions': {
                                        'format': 'png',
                                        'filename': 'prediction_population_mondiale',
                                        'height': 700,
                                        'width': 1200,
                                        'scale': 2
                                    }
                                },
                                style={'height': '100%'}
                            )
                        ], className='card-body')
                    ], className='card')
                ], style={'height': 'calc(80vh - 160px)', 'marginBottom': '20px'}),
                
                # Tableau des prédictions
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3('Tableau des Prédictions pour 2050', 
                                  style={'margin-top': '0', 'font-size': '16px'})
                        ], className='card-header'),
                        
                        html.Div([
                            dash_table.DataTable(
                                data=[{
                                    'Modèle': model.replace('_', ' ').title(),
                                    'Population en 2050 (milliards)': f"{predictions_df[predictions_df['year'] == 2050][model].values[0]:.3f}",
                                    'Croissance depuis 2023 (%)': f"{((predictions_df[predictions_df['year'] == 2050][model].values[0] / last_pop) - 1) * 100:.1f}%"
                                } for model in ['linear_regression', 'random_forest', 'xgboost', 'arima']],
                                columns=[
                                    {'name': 'Modèle', 'id': 'Modèle'},
                                    {'name': 'Population en 2050 (milliards)', 'id': 'Population en 2050 (milliards)'},
                                    {'name': 'Croissance depuis 2023 (%)', 'id': 'Croissance depuis 2023 (%)'}
                                ],
                                style_table={'width': '100%'},
                                style_cell={
                                    'font-family': '"Segoe UI", Arial, sans-serif',
                                    'padding': '10px 15px',
                                    'textAlign': 'center'
                                },
                                style_header={
                                    'backgroundColor': blue_theme['accent'],
                                    'color': 'white',
                                    'fontWeight': 'bold',
                                    'textAlign': 'center'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': '#f8fafd'
                                    }
                                ]
                            )
                        ], className='card-body table-container')
                    ], className='card')
                ], style={'height': 'calc(20vh)'})
            ])
        ])

# Callback pour mettre à jour la carte
@app.callback(
    Output('map-graph', 'figure'),
    [Input('map-variable', 'value'),
     Input('year-slider', 'value')]
)
def update_map(selected_var, selected_year):
    # Utiliser des valeurs par défaut si nécessaire
    selected_var = selected_var if selected_var is not None else 'population_growth_annual_pct'
    selected_year = selected_year if selected_year is not None else 2020
    
    # Filtrer les données pour l'année sélectionnée
    filtered_df = df[df['year'] == selected_year].copy()
    
    # Exclure les agrégats régionaux et garder uniquement les pays
    country_df = filtered_df[~filtered_df['country_code'].isin(regions)].copy()
    
    # Créer la carte avec Plotly
    var_title = selected_var.replace('_', ' ').title()
    
    if selected_var == 'population_growth_annual_pct':
        color_scale = [
            [0, 'rgb(178,24,43)'],    # Rouge foncé pour les valeurs négatives
            [0.33, 'rgb(239,138,98)'], # Rouge clair pour les valeurs faiblement négatives
            [0.5, 'rgb(253,219,199)'], # Beige pour les valeurs proches de zéro
            [0.67, 'rgb(209,229,240)'], # Bleu clair pour les valeurs faiblement positives
            [1, 'rgb(33,102,172)']     # Bleu foncé pour les valeurs fortement positives
        ]
        range_color = [-2, 4]  # Ajuster selon les données
    elif selected_var == 'gdp_per_capita_current_usd':
        color_scale = 'Blues'
        range_color = [0, 70000]  # Ajuster selon les données
    else:
        color_scale = 'Viridis'
        range_color = None
    
    fig = px.choropleth(
        country_df,
        locations='country_code',
        color=selected_var,
        hover_name='country_name',
        color_continuous_scale=color_scale,
        range_color=range_color,
        labels={selected_var: var_title},
        template='plotly_white'
    )
    
    fig.update_layout(
        geo=dict(
            showframe=True,
            showcoastlines=True,
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(250, 250, 250)',
            coastlinecolor='rgb(180, 180, 180)',
            countrycolor='rgb(180, 180, 180)',
            showocean=True,
            oceancolor='rgb(240, 247, 255)'
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title=var_title,
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            yanchor="top", y=1,
            ticks="outside",
            dtick=1,
            outlinewidth=1,
            outlinecolor='rgba(0,0,0,0.2)'
        ),
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Amélioration des traces
    for trace in fig.data:
        # Augmenter l'épaisseur des lignes et le contraste des couleurs
        if 'line' in trace:
            # Rendre la couleur plus vive en ajustant la saturation
            color = trace.line.color
            if color.startswith('rgb'):
                r, g, b = map(int, color.replace('rgb(', '').replace(')', '').split(','))
                # Ajuster la saturation pour plus de contraste
                max_val = max(r, g, b)
                if max_val > 0:
                    ratio = 255 / max_val
                    r = min(255, int(r * ratio * 0.9))
                    g = min(255, int(g * ratio * 0.9))
                    b = min(255, int(b * ratio * 0.9))
                    trace.line.color = f'rgb({r},{g},{b})'
            
            trace.line.width = trace.line.width * 1.2 if hasattr(trace.line, 'width') else 3
    
    return fig

# Callback pour mettre à jour le graphique de tendance
@app.callback(
    Output('trend-graph', 'figure'),
    [Input('trend-variable', 'value')]
)
def update_trend_graph(selected_var):
    # Utiliser une valeur par défaut si nécessaire
    selected_var = selected_var if selected_var is not None else 'population_growth_annual_pct'
    
    # Création du graphique pour les grands pays
    fig = go.Figure()
    
    # Palette de couleurs pour les différents pays
    colors = px.colors.qualitative.D3
    
    for i, country_code in enumerate(major_countries):
        country_data = df[df['country_code'] == country_code]
        if not country_data.empty:
            country_name = country_data['country_name'].iloc[0]
            
            fig.add_trace(go.Scatter(
                x=country_data['year'], 
                y=country_data[selected_var],
                mode='lines+markers',
                name=country_name,
                line=dict(width=2, color=colors[i % len(colors)]),
                marker=dict(size=6, color=colors[i % len(colors)])
            ))
    
    var_title = selected_var.replace('_', ' ').title()
    
    fig.update_layout(
        xaxis=dict(
            title="Année",
            gridcolor='rgba(220, 227, 234, 0.7)',
            tickfont=dict(family="Segoe UI, sans-serif", size=11),
            zeroline=False,
            titlefont=dict(family="Segoe UI, sans-serif", size=13)
        ),
        yaxis=dict(
            title=var_title,
            gridcolor='rgba(220, 227, 234, 0.7)',
            tickfont=dict(family="Segoe UI, sans-serif", size=11),
            zeroline=False,
            titlefont=dict(family="Segoe UI, sans-serif", size=13)
        ),
        legend=dict(
            font=dict(family="Segoe UI, sans-serif", size=11),
            orientation="h",
            y=-0.15,
            yanchor="top",
            x=0.5,
            xanchor="center",
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.05)',
            borderwidth=1
        ),
        hovermode="closest",
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=20, t=10, b=60),
        height=350
    )
    
    # Amélioration des traces pour un style plus moderne
    for trace in fig.data:
        trace.update(
            line=dict(width=3),
            marker=dict(size=8, line=dict(width=1, color='white'))
        )
    
    # Amélioration des traces
    for trace in fig.data:
        # Augmenter l'épaisseur des lignes et le contraste des couleurs
        if 'line' in trace:
            # Rendre la couleur plus vive en ajustant la saturation
            color = trace.line.color
            if color.startswith('rgb'):
                r, g, b = map(int, color.replace('rgb(', '').replace(')', '').split(','))
                # Ajuster la saturation pour plus de contraste
                max_val = max(r, g, b)
                if max_val > 0:
                    ratio = 255 / max_val
                    r = min(255, int(r * ratio * 0.9))
                    g = min(255, int(g * ratio * 0.9))
                    b = min(255, int(b * ratio * 0.9))
                    trace.line.color = f'rgb({r},{g},{b})'
            
            trace.line.width = trace.line.width * 1.2 if hasattr(trace.line, 'width') else 3
    
    return fig

# Callback pour mettre à jour le tableau de résumé
@app.callback(
    Output('summary-table', 'data'),
    Output('summary-table', 'columns'),
    [Input('map-variable', 'value'),
     Input('year-slider', 'value')]
)
def update_summary_table(selected_var, selected_year):
    # Utiliser des valeurs par défaut si nécessaire
    selected_var = selected_var if selected_var is not None else 'population_growth_annual_pct'
    selected_year = selected_year if selected_year is not None else 2020
    
    # Filtrer les données pour les grands pays et l'année sélectionnée
    filtered_df = df[(df['country_code'].isin(major_countries)) & 
                     (df['year'] == selected_year)].copy()
    
    # Sélectionner les colonnes pertinentes
    display_columns = ['country_name', selected_var]
    
    # Créer le tableau
    table_data = filtered_df[display_columns].sort_values(by=selected_var, ascending=False)
    
    # Formater les données pour l'affichage
    formatted_data = []
    for _, row in table_data.iterrows():
        formatted_row = {'Pays': row['country_name']}
        
        if selected_var == 'population_growth_annual_pct':
            value = row[selected_var]
            formatted_row['Valeur'] = f"{value:.2f}%" if pd.notna(value) else "N/A"
        elif selected_var == 'gdp_per_capita_current_usd':
            value = row[selected_var]
            formatted_row['Valeur'] = f"${value:,.0f}" if pd.notna(value) else "N/A"
        else:
            value = row[selected_var]
            formatted_row['Valeur'] = f"{value:.2f}" if pd.notna(value) else "N/A"
        
        formatted_data.append(formatted_row)
    
    # Définir les colonnes
    columns = [
        {'name': 'Pays', 'id': 'Pays'},
        {'name': selected_var.replace('_', ' ').title(), 'id': 'Valeur'}
    ]
    
    return formatted_data, columns

# Callback pour mettre à jour la comparaison régionale
@app.callback(
    Output('region-comparison', 'figure'),
    [Input('trend-variable', 'value')]
)
def update_region_comparison(selected_var):
    # Utiliser une valeur par défaut si nécessaire
    selected_var = selected_var if selected_var is not None else 'population_growth_annual_pct'
    
    # Filtrer les données pour les régions
    region_df = df[df['country_code'].isin(regions)].copy()
    
    # Créer le graphique pour les régions
    fig = go.Figure()
    
    # Palette de couleurs pour les différentes régions
    colors = px.colors.qualitative.Bold
    
    region_names = {
        'WLD': 'Monde',
        'EAS': 'Asie de l\'Est & Pacifique',
        'ECS': 'Europe & Asie centrale',
        'LCN': 'Amérique latine & Caraïbes',
        'MEA': 'Moyen-Orient & Afrique du Nord',
        'NAC': 'Amérique du Nord',
        'SAS': 'Asie du Sud',
        'SSF': 'Afrique subsaharienne'
    }
    
    for i, region_code in enumerate(regions):
        region_data = region_df[region_df['country_code'] == region_code]
        if not region_data.empty:
            region_name = region_names.get(region_code, region_data['country_name'].iloc[0])
            
            fig.add_trace(go.Scatter(
                x=region_data['year'], 
                y=region_data[selected_var],
                mode='lines',
                name=region_name,
                line=dict(width=2.5, color=colors[i % len(colors)])
            ))
    
    var_title = selected_var.replace('_', ' ').title()
    
    fig.update_layout(
        xaxis=dict(
            title="Année",
            gridcolor='rgba(220, 227, 234, 0.7)',
            tickfont=dict(family="Segoe UI, sans-serif", size=11),
            zeroline=False,
            titlefont=dict(family="Segoe UI, sans-serif", size=13)
        ),
        yaxis=dict(
            title=var_title,
            gridcolor='rgba(220, 227, 234, 0.7)',
            tickfont=dict(family="Segoe UI, sans-serif", size=11),
            zeroline=False,
            titlefont=dict(family="Segoe UI, sans-serif", size=13)
        ),
        legend=dict(
            font=dict(family="Segoe UI, sans-serif", size=11),
            orientation="h",
            y=-0.15,
            yanchor="top",
            x=0.5,
            xanchor="center",
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.05)',
            borderwidth=1
        ),
        hovermode="closest",
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=20, t=10, b=60),
        height=350
    )
    
    # Amélioration des traces
    for trace in fig.data:
        trace.update(line=dict(width=3.5))
    
    # Amélioration des traces
    for trace in fig.data:
        # Augmenter l'épaisseur des lignes et le contraste des couleurs
        if 'line' in trace:
            # Rendre la couleur plus vive en ajustant la saturation
            color = trace.line.color
            if color.startswith('rgb'):
                r, g, b = map(int, color.replace('rgb(', '').replace(')', '').split(','))
                # Ajuster la saturation pour plus de contraste
                max_val = max(r, g, b)
                if max_val > 0:
                    ratio = 255 / max_val
                    r = min(255, int(r * ratio * 0.9))
                    g = min(255, int(g * ratio * 0.9))
                    b = min(255, int(b * ratio * 0.9))
                    trace.line.color = f'rgb({r},{g},{b})'
            
            trace.line.width = trace.line.width * 1.2 if hasattr(trace.line, 'width') else 3
    
    return fig

# Callback pour mettre à jour le graphique de prédiction
@app.callback(
    Output('prediction-graph', 'figure'),
    [Input('tabs', 'value')]  # Même si on n'utilise pas cette entrée, cela force le rappel
)
def update_prediction_graph(_):
    # Créer le graphique de prédiction
    fig = go.Figure()
    
    # Définir la palette de couleurs
    colors = {
        'historical': blue_theme['secondary'],
        'linear': '#E41A1C',  # Rouge
        'rf': '#4DAF4A',      # Vert
        'xgb': '#984EA3',     # Violet
        'arima': '#FF7F00'    # Orange
    }
    
    # Données historiques
    fig.add_trace(go.Scatter(
        x=population_data['year'],
        y=population_data['population_billions'],
        mode='lines+markers',
        name='Données historiques',
        line=dict(color=colors['historical'], width=3),
        marker=dict(size=8, color=colors['historical'])
    ))
    
    # Prédictions
    models = {
        'linear_regression': {'name': 'Régression Linéaire', 'color': colors['linear'], 'dash': 'dash'},
        'random_forest': {'name': 'Random Forest', 'color': colors['rf'], 'dash': 'dot'},
        'xgboost': {'name': 'XGBoost', 'color': colors['xgb'], 'dash': 'dashdot'},
        'arima': {'name': 'ARIMA', 'color': colors['arima'], 'dash': 'solid'}
    }
    
    for model, style in models.items():
        fig.add_trace(go.Scatter(
            x=predictions_df['year'],
            y=predictions_df[model],
            mode='lines',
            name=style['name'],
            line=dict(color=style['color'], width=2.5, dash=style['dash'])
        ))
    
    # Ajouter une ligne verticale pour marquer le début des prédictions
    current_year = population_data['year'].max()
    
    fig.add_shape(
        type="line",
        x0=current_year,
        x1=current_year,
        y0=0,
        y1=12,  # Ajuster en fonction de l'échelle
        line=dict(color="gray", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=current_year + 2,
        y=population_data[population_data['year'] == current_year]['population_billions'].values[0] + 0.5,
        text=f"Début des prédictions ({current_year})",
        showarrow=True,
        arrowhead=1,
        font=dict(size=12, color='gray')
    )
    
    # Ajouter des annotations pour les valeurs finales en 2050
    for model, style in models.items():
        value_2050 = predictions_df[predictions_df['year'] == 2050][model].values[0]
        fig.add_annotation(
            x=2050,
            y=value_2050,
            text=f"{value_2050:.2f}",
            showarrow=False,
            font=dict(size=10, color=style['color']),
            xanchor='left',
            xshift=5
        )
    
    fig.update_layout(
        xaxis=dict(
            title="Année",
            gridcolor='rgba(220, 227, 234, 0.7)',
            tickfont=dict(family="Segoe UI, sans-serif", size=12),
            zeroline=False,
            titlefont=dict(family="Segoe UI, sans-serif", size=14),
            range=[1980, 2055]  # Étendre légèrement pour les annotations
        ),
        yaxis=dict(
            title="Population (milliards)",
            gridcolor='rgba(220, 227, 234, 0.7)',
            tickfont=dict(family="Segoe UI, sans-serif", size=12),
            zeroline=False,
            titlefont=dict(family="Segoe UI, sans-serif", size=14)
        ),
        legend=dict(
            font=dict(family="Segoe UI, sans-serif", size=12),
            orientation="h",
            y=1.12,
            yanchor="bottom",
            x=0.5,
            xanchor="center",
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.05)',
            borderwidth=1
        ),
        hovermode="x unified",
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=30, b=60),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Segoe UI, sans-serif",
            bordercolor="rgba(0,0,0,0.1)"
        )
    )
    
    # Mettre en valeur la zone de prédiction
    fig.add_shape(
        type="rect",
        x0=current_year,
        x1=2050,
        y0=0,
        y1=12,
        fillcolor="rgba(220, 230, 240, 0.2)",
        line=dict(width=0),
        layer="below"
    )
    
    # Amélioration du style des traces
    for i, trace in enumerate(fig.data):
        if i == 0:  # Données historiques
            trace.update(
                line=dict(width=4, color=colors['historical']),
                marker=dict(size=9, line=dict(width=1.5, color='white'))
            )
        else:  # Prédictions
            trace.update(line=dict(width=3.5))
    
    # Amélioration des traces
    for trace in fig.data:
        # Augmenter l'épaisseur des lignes et le contraste des couleurs
        if 'line' in trace:
            # Rendre la couleur plus vive en ajustant la saturation
            color = trace.line.color
            if color.startswith('rgb'):
                r, g, b = map(int, color.replace('rgb(', '').replace(')', '').split(','))
                # Ajuster la saturation pour plus de contraste
                max_val = max(r, g, b)
                if max_val > 0:
                    ratio = 255 / max_val
                    r = min(255, int(r * ratio * 0.9))
                    g = min(255, int(g * ratio * 0.9))
                    b = min(255, int(b * ratio * 0.9))
                    trace.line.color = f'rgb({r},{g},{b})'
            
            trace.line.width = trace.line.width * 1.2 if hasattr(trace.line, 'width') else 3
    
    return fig

# Fonction pour ouvrir automatiquement le navigateur
def open_browser():
    """Ouvre le navigateur par défaut une seule fois."""
    global _browser_opened
    if '_browser_opened' not in globals():
        _browser_opened = True
        time.sleep(1)
        webbrowser.open_new("http://127.0.0.1:8050/")
        
        
# Lancer l'application
if __name__ == '__main__':
    # Récupérer le port depuis l'environnement (pour Render) ou utiliser 8050 par défaut
    port = int(os.environ.get('PORT', 8050))
    
    # Démarrer le thread du navigateur seulement en local
    if os.environ.get('RENDER') != 'true':
        threading.Thread(target=open_browser).start()
    
    # Toujours écouter sur 0.0.0.0 (toutes les interfaces) pour que Render puisse détecter le port
    print(f"Démarrage de l'application sur le port {port}")
    app.run_server(
        host='0.0.0.0',  # Nécessaire pour Render
        port=port,
        debug=False
    )