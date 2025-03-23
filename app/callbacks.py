"""
Module pour la gestion des callbacks Dash.
Ce module contient les fonctions qui définissent le comportement interactif
de l'application en réponse aux actions utilisateur.
"""

import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
import os

# Import des données et fonctions utilitaires
from data import data, CONTINENT_COLORS, filter_data_by_year_range, get_correlation_stats

# Définition des callbacks de l'application
def register_callbacks(app):
    """
    Enregistre tous les callbacks pour l'application.
    
    Args:
        app: L'instance de l'application Dash
    """
    
    # Callback pour mettre à jour l'affichage de la plage d'années sélectionnée
    @app.callback(
        Output('year-range-display', 'children'),
        [Input('year-slider', 'value')]
    )
    def update_year_range_display(year_range):
        """
        Met à jour l'affichage de la plage d'années sélectionnée.
        
        Args:
            year_range (list): Liste [année_min, année_max]
            
        Returns:
            list: Elements HTML pour l'affichage
        """
        return [
            html.Span(f"{year_range[0]}", className="year-start"),
            html.Span(" - "),
            html.Span(f"{year_range[1]}", className="year-end")
        ]
    
    # Callback pour mettre à jour la carte choroplèthe (graphique en haut à gauche)
    @app.callback(
        [Output('choropleth-map', 'figure'), Output('choropleth-title', 'children')],
        [
            Input('year-slider', 'value'),
            Input('x-variable-dropdown', 'value')
        ]
    )
    def update_choropleth(year_range, x_variable):
        """
        Met à jour la carte choroplèthe en fonction de la variable sélectionnée.
        
        Args:
            year_range (list): Liste [année_min, année_max]
            x_variable (str): Variable pour la carte
            
        Returns:
            Figure: La figure de la carte mise à jour
        """
        # Créer une figure vide par défaut (au cas où une erreur se produirait)
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            x=0.5, y=0.5,
            text="Aucune donnée disponible",
            showarrow=False,
            font=dict(color="white", size=14)
        )
        empty_fig.update_layout(
            paper_bgcolor="#0c1445",
            plot_bgcolor="#0c1445",
            font=dict(color="white"),
            height=400
        )
    
        try:
            # Vérifier que la variable est bien spécifiée
            if not x_variable:
                return empty_fig, "Aucune variable sélectionnée"
                
            # Filtrer les données par plage d'années
            filtered_data = filter_data_by_year_range(data, year_range[0], year_range[1])
            
            # Au lieu de calculer la moyenne, prenons la dernière année de la plage
            last_year = year_range[1]
            last_year_data = filtered_data[filtered_data['Year'] == last_year]
            
            # Si aucune donnée n'est disponible pour la dernière année, prendre l'année la plus récente disponible
            if last_year_data.empty:
                available_years = filtered_data['Year'].unique()
                if len(available_years) > 0:
                    last_year = max(available_years)
                    last_year_data = filtered_data[filtered_data['Year'] == last_year]
                else:
                    # Si toujours aucune donnée, utiliser toutes les données filtrées
                    last_year_data = filtered_data
            
            # Extraire les valeurs par pays pour la dernière année
            # Assurons-nous que x_variable existe dans le dataset
            if x_variable not in last_year_data.columns:
                # Créer une figure vide avec un message d'erreur
                fig = go.Figure()
                fig.add_annotation(
                    x=0.5, y=0.5,
                    text=f"Variable {x_variable} non disponible",
                    showarrow=False,
                    font=dict(color="white", size=14)
                )
                fig.update_layout(
                    paper_bgcolor="#0c1445",
                    plot_bgcolor="#0c1445",
                    font=dict(color="white"),
                    height=400
                )
                return fig, f"Erreur: {x_variable} non disponible"
            
            # Nettoyer les données pour la carte
            country_data = last_year_data.dropna(subset=[x_variable, 'Country Code'])
            
            # S'assurer qu'il y a des données à afficher
            if country_data.empty:
                # Créer une figure vide avec un message
                fig = go.Figure()
                fig.add_annotation(
                    x=0.5, y=0.5,
                    text="Aucune donnée disponible pour cette sélection",
                    showarrow=False,
                    font=dict(color="white", size=14)
                )
                fig.update_layout(
                    paper_bgcolor="#0c1445",
                    plot_bgcolor="#0c1445",
                    font=dict(color="white"),
                    height=400
                )
                return fig, f"Pas de données ({last_year})"
            
            # Créer un DataFrame simplifié avec les données nécessaires pour la carte
            plot_data = country_data[['Country Name', 'Country Code', 'Continent', x_variable]].drop_duplicates()
            
            # Convertir les valeurs infinies et remplacer les NaN
            plot_data = plot_data.replace([np.inf, -np.inf], np.nan)
            plot_data = plot_data.dropna(subset=[x_variable])
            
            # S'assurer que le dataframe n'est pas vide après le nettoyage
            if plot_data.empty:
                return empty_fig, f"Pas de données valides ({last_year})"
            
            # Vérifier les codes pays pour s'assurer qu'ils sont valides
            # Les codes ISO-3 doivent avoir 3 caractères
            plot_data = plot_data[plot_data['Country Code'].str.len() == 3]
            
            # S'assurer que le dataframe n'est pas vide après le filtrage des codes pays
            if plot_data.empty:
                return empty_fig, f"Pas de codes pays valides ({last_year})"
            
            # Utiliser un choropleth de plotly.graph_objects pour un contrôle total
            fig = go.Figure(data=go.Choropleth(
                locations=plot_data['Country Code'],
                z=plot_data[x_variable],
                text=plot_data['Country Name'],
                colorscale='Magma',
                autocolorscale=False,
                marker_line_color='rgba(255, 255, 255, 0.2)',
                marker_line_width=0.5,
                colorbar=dict(
                    title=None,
                    titleside='right',
                    tickfont=dict(color='rgba(0,0,0,0)'),
                    titlefont=dict(color='rgba(0,0,0,0)'),
                    bgcolor='rgba(0,0,0,0)',
                    tickformat='.3f',
                    len=0
                ),
                hovertemplate='<b>%{text}</b><br>%{z:.3f}<extra></extra>',
                locationmode='ISO-3',
                showscale=False
            ))
            
            # Personnalisation du layout avec fond transparent et optimisé pour "Répartition mondiale"
            fig.update_layout(
                title=None,  # Supprimer le titre car nous l'affichons dans le header
                geo=dict(
                    scope='world',
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor='rgba(255, 255, 255, 0.3)',
                    showland=True,
                    landcolor='rgba(19, 33, 88, 0.5)',  # Ajusté pour une meilleure visibilité
                    showocean=True,
                    oceancolor='rgba(12, 20, 69, 0.5)',  # Ajusté pour une meilleure visibilité
                    showlakes=True,
                    lakecolor='rgba(12, 20, 69, 0.5)',  # Ajusté pour une meilleure visibilité
                    showcountries=True,
                    countrycolor='rgba(255, 255, 255, 0.2)',
                    projection_type='natural earth',
                    bgcolor='rgba(0,0,0,0)',
                    lonaxis=dict(
                        showgrid=False,  # Supprimer la grille de longitude
                        gridcolor='rgba(0,0,0,0)'
                    ),
                    lataxis=dict(
                        showgrid=False,  # Supprimer la grille de latitude
                        gridcolor='rgba(0,0,0,0)'
                    )
                ),
                paper_bgcolor='rgba(0,0,0,0)',  # Fond transparent
                plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
                margin=dict(l=0, r=0, t=0, b=0),
                height=400,
                font=dict(color='white')
            )
            
            # Retourner la figure et le titre pour l'affichage à côté de "Répartition mondiale"
            return fig, f"{x_variable} ({last_year})"
        
        except Exception as e:
            # En cas d'erreur, afficher une figure vide avec un message d'erreur
            print(f"Erreur dans update_choropleth: {str(e)}")
            
            error_fig = go.Figure()
            error_fig.add_annotation(
                x=0.5, y=0.5,
                text=f"Une erreur s'est produite",
                showarrow=False,
                font=dict(color="white", size=14)
            )
            error_fig.update_layout(
                paper_bgcolor="#0c1445",
                plot_bgcolor="#0c1445",
                font=dict(color="white"),
                height=400
            )
            
            return error_fig, "Erreur"
    
    # Callback pour mettre à jour le violin plot (graphique en bas à gauche)
    @app.callback(
        Output('violin-plot', 'figure'),
        [
            Input('year-slider', 'value'),
            Input('x-variable-dropdown', 'value')
        ]
    )
    def update_violin_plot(year_range, x_variable):
        """
        Met à jour le violin plot d'espérance de vie par continent.
        
        Args:
            year_range (list): Liste [année_min, année_max]
            x_variable (str): Variable pour le violin plot
            
        Returns:
            Figure: La figure du violin plot mise à jour
        """
        # Filtrer les données par plage d'années
        filtered_data = filter_data_by_year_range(data, year_range[0], year_range[1])
        # Utiliser tous les continents
        continents = ['Africa', 'North America', 'South America', 'Asia', 'Europe', 'Oceania']
        filtered_data = filtered_data[filtered_data["Continent"].isin(continents)]
        
        # Création du violin plot
        fig = px.violin(
            filtered_data,
            x='Continent',
            y=x_variable,
            box=False,  # Désactiver la boîte à moustaches au centre
            points=False,
            color="Continent",
            color_discrete_map=CONTINENT_COLORS,
            title=f"Distribution de {x_variable} par continent",
            # Format personnalisé pour les tooltips
            hover_data={x_variable: ':.3f'}
        )
        
        # Ajustement du layout
        fig.update_layout(
            title=None,  # Suppression du titre pour économiser de l'espace
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                color='white'
            ),
            xaxis=dict(
                title=dict(
                    text='Continent',
                    standoff=10  # Ajouter un espace entre le titre et l'axe
                ),
                tickfont=dict(size=10),
                gridcolor='rgba(255,255,255,0.05)'  # Grille presque invisible
            ),
            yaxis=dict(
                title=dict(
                    text=x_variable
                ),
                gridcolor='rgba(0,0,0,0)',  # Suppression complète de la grille horizontale
                zeroline=True,              # Garder la ligne zéro
                zerolinecolor='rgba(255,255,255,0.2)',  # Couleur de la ligne zéro
                zerolinewidth=1,            # Épaisseur de la ligne zéro
                showgrid=False,             # Supprimer les traits horizontaux de l'axe Y
                # Désactiver la notation scientifique pour l'axe Y
                tickformat='.3f',
                exponentformat='none'
            ),
            legend=dict(
                orientation="h",            # Orientation horizontale
                yanchor="bottom",           # Ancrage en bas
                y=1.02,                     # Position au-dessus du graphique
                xanchor="center",           # Ancrage au centre
                x=0.5,                      # Centré
                bgcolor="rgba(0,0,0,0.1)",  # Fond légèrement transparent
                font=dict(color="white")    # Texte blanc
            ),
            margin=dict(l=50, r=20, t=30, b=70),  # Augmenter la marge du bas pour le titre de l'axe x
            hoverlabel=dict(
                bgcolor='rgba(19,33,88,0.8)',  # Fond de l'infobulle avec légère transparence
                bordercolor='#0072ff',  # Bordure de l'infobulle
                font=dict(color='white', size=10)  # Texte de l'infobulle
            )
        )
        
        # Modifier les lignes du violin plot pour être aux couleurs des continents
        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'line'):
                trace.line.color = list(CONTINENT_COLORS.values())[i % len(CONTINENT_COLORS)]
                trace.line.width = 1.5
            # Modifier le format du template du tooltip
            if hasattr(trace, 'hovertemplate'):
                trace.hovertemplate = trace.hovertemplate.replace('%{y}', '%{y:.3f}')
        
        return fig
    
    # Callback pour mettre à jour le line plot (graphique en haut à droite)
    @app.callback(
        Output('line-plot', 'figure'),
        [
            Input('year-slider', 'value'),
            Input('y-variable-dropdown', 'value')
        ]
    )
    def update_line_plot(year_range, y_variable):
        """
        Met à jour le line plot d'évolution de la variable par continent.
        
        Args:
            year_range (list): Liste [année_min, année_max]
            y_variable (str): Variable pour le line plot
            
        Returns:
            Figure: La figure du line plot mise à jour
        """
        # Filtrer les données par plage d'années
        filtered_data = filter_data_by_year_range(data, year_range[0], year_range[1])
        
        # Utiliser tous les continents
        # Modifier pour prendre en compte North America et South America séparément
        continents = ['Africa', 'North America', 'South America', 'Asia', 'Europe', 'Oceania']
        filtered_data = filtered_data[filtered_data["Continent"].isin(continents)]
        
        # Calcul de la somme (ou moyenne selon la variable) par continent et par année
        is_population = 'Population' in y_variable
        
        # Pour les variables démographiques, on utilise la somme
        if is_population:
            agg_df = filtered_data.groupby(['Year', 'Continent'])[y_variable].sum().reset_index()
            # Conversion en milliards pour toutes les variables de population
            agg_df[y_variable] = agg_df[y_variable] / 1_000_000_000
            y_axis_title = f"{y_variable} (milliards)"
        else:
            # Pour les autres variables (comme espérance de vie, taux, etc.), on utilise la moyenne pondérée
            # Créer un DataFrame temporaire avec les calculs
            temp_df = filtered_data.copy()
            grouped = temp_df.groupby(['Year', 'Continent']).apply(
                lambda x: np.average(
                    x[y_variable], 
                    weights=x['Population, total'] if 'Population, total' in x.columns else None
                )
            ).reset_index(name=y_variable)
            agg_df = grouped
            y_axis_title = y_variable
        
        # Création du graphique avec lignes simples (sans aires, sans points)
        fig = go.Figure()
        
        # Ajout des lignes pour chaque continent
        for continent in continents:
            if continent in agg_df['Continent'].unique():
                continent_data = agg_df[agg_df['Continent'] == continent]
                fig.add_trace(go.Scatter(
                    x=continent_data['Year'],
                    y=continent_data[y_variable],
                    name=continent,
                    mode='lines',  # Seulement des lignes - pas de points
                    line=dict(
                        color=CONTINENT_COLORS.get(continent, '#636EFA'),
                        width=2
                    ),
                    # Format personnalisé pour les tooltips
                    hovertemplate=f"{continent}<br>{y_axis_title}: %{{y:.3f}}<extra></extra>"
                ))
        
        # Ajustement du layout
        fig.update_layout(
            title=None,  # Suppression du titre pour économiser de l'espace
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                color='white'
            ),
            xaxis=dict(
                title=dict(
                    text='Année',
                    standoff=10  # Ajouter un espace entre le titre et l'axe
                ),
                tickmode='linear',
                dtick=5,
                gridcolor='rgba(255,255,255,0.05)'  # Grille presque invisible
            ),
            yaxis=dict(
                title=dict(
                    text=y_axis_title
                ),
                gridcolor='rgba(255,255,255,0.05)',  # Grille presque invisible
                # Désactiver la notation scientifique pour l'axe Y
                tickformat='.3f',  # Format à 3 décimales
                exponentformat='none'  # Désactiver le format scientifique
            ),
            legend=dict(
                orientation="h",        
                yanchor="top",              # Ancrage en haut
                y=0.99,                     # Position près du haut
                xanchor="right",            # Ancrage à droite
                x=0.99,                     # Position près de la droite
                bgcolor="rgba(0,0,0,0.1)",  # Fond légèrement transparent
                font=dict(size=9, color="white")
            ),
            margin=dict(l=50, r=20, t=20, b=60),  # Marges ajustées
            hovermode="x unified",  # Hover qui montre tous les continents sur un même x
            hoverlabel=dict(
                bgcolor='rgba(19,33,88,0.8)',  # Fond de l'infobulle avec légère transparence
                bordercolor='#0072ff',  # Bordure de l'infobulle
                font=dict(color='white', size=10)  # Texte de l'infobulle
            )
        )
        
        return fig
    
    # Callback pour mettre à jour le scatter plot (graphique en bas à droite)
    @app.callback(
        [Output('scatter-plot', 'figure'), Output('correlation-stats', 'children')],
        [
            Input('year-slider', 'value'),
            Input('x-variable-dropdown', 'value'),
            Input('y-variable-dropdown', 'value')
        ]
    )
    def update_scatter_plot(year_range, x_variable, y_variable):
        """
        Met à jour le scatter plot et les statistiques de corrélation.
        
        Args:
            year_range (list): Liste [année_min, année_max]
            x_variable (str): Variable pour l'axe X
            y_variable (str): Variable pour l'axe Y
            
        Returns:
            Figure: La figure du scatter plot mise à jour
        """
        # Créer un graphique vide par défaut (au cas où une erreur se produirait)
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            x=0.5, y=0.5,
            text="Aucune donnée disponible",
            showarrow=False,
            font=dict(color="white", size=14)
        )
        empty_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        
        # Message d'erreur pour les statistiques
        error_stats = [
            html.Div("Aucune donnée disponible", className="stat-item error")
        ]
        
        try:
            # Vérifier que les variables sont bien spécifiées
            if not x_variable or not y_variable:
                return empty_fig, error_stats
                
            # Filtrer les données par plage d'années
            filtered_data = filter_data_by_year_range(data, year_range[0], year_range[1])
            
            # Utiliser tous les continents - en cohérence avec les autres graphiques
            continents = ['Africa', 'North America', 'South America', 'Asia', 'Europe', 'Oceania']
            filtered_data = filtered_data[filtered_data["Continent"].isin(continents)]
            
            # Supprimer les valeurs infinies
            filtered_data = filtered_data.replace([np.inf, -np.inf], np.nan)
            
            # Supprimer les lignes avec des valeurs manquantes dans les variables X et Y
            filtered_data = filtered_data.dropna(subset=[x_variable, y_variable])
            
            # Éviter les erreurs si le DataFrame est vide après filtrage
            if filtered_data.empty:
                return empty_fig, error_stats
            
            # Calculer la moyenne des variables pour chaque pays sur la période sélectionnée
            try:
                country_avg = filtered_data.groupby(["Country Name", "Continent"]).agg({
                    x_variable: "mean",
                    y_variable: "mean",
                    "Population (Millions)": "mean"
                }).reset_index()
            except Exception as e:
                print(f"Erreur dans l'agrégation par pays: {e}")
                return empty_fig, error_stats
            
            # Vérifier que le DataFrame agrégé n'est pas vide
            if country_avg.empty:
                return empty_fig, error_stats
            
            # Créer le scatter plot
            fig = px.scatter(
                country_avg,
                x=x_variable,
                y=y_variable,
                color="Continent",
                color_discrete_map=CONTINENT_COLORS,
                hover_name="Country Name",
                size="Population (Millions)",
                size_max=25,  # Légèrement plus petit pour économiser de l'espace
                opacity=0.8,
                title=f"Relation entre {x_variable} et {y_variable}",
                # Configurer le format des tooltips pour éviter les notations scientifiques
                hover_data={
                    x_variable: ':.3f',
                    y_variable: ':.3f',
                    "Population (Millions)": ':.2f'
                }
            )
            
            # Calculer la droite de régression
            correlation_stats = get_correlation_stats(country_avg, x_variable, y_variable)
            
            # Ajouter la droite de régression si la corrélation est significative
            if abs(correlation_stats["correlation"]) > 0.1:
                try:
                    x_range = np.linspace(
                        country_avg[x_variable].min(),
                        country_avg[x_variable].max(),
                        100
                    )
                    y_range = correlation_stats["slope"] * x_range + correlation_stats["intercept"]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_range,
                            mode="lines",
                            name="Régression",
                            line=dict(color="rgba(255,255,255,0.5)", width=2)
                        )
                    )
                except Exception as e:
                    print(f"Erreur dans le calcul de la droite de régression: {e}")
                    # Continuer sans la ligne de régression
            
            # Ajustement du layout
            fig.update_layout(
                title=None,  # Suppression du titre pour économiser de l'espace
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(
                    color='white'
                ),
                xaxis=dict(
                    title=dict(
                        text=x_variable,
                        standoff=15  # Éloigner le titre de l'axe
                    ),
                    tickfont=dict(size=9),  # Réduire la taille des étiquettes
                    automargin=True,  # Ajuster automatiquement les marges
                    gridcolor='rgba(0,0,0,0)',  # Grille transparente (supprimée)
                    # Désactiver la notation scientifique pour l'axe X
                    tickformat='.3f',
                    exponentformat='none'
                ),
                yaxis=dict(
                    title=dict(
                        text=y_variable
                    ),
                    gridcolor='rgba(0,0,0,0)',  # Grille transparente (supprimée)
                    # Désactiver la notation scientifique pour l'axe Y
                    tickformat='.3f',
                    exponentformat='none'
                ),
                legend=dict(
                    orientation="h",         # Orientation verticale
                    yanchor="top",           # Ancrage en haut
                    y=0.99,                  # Position près du haut
                    xanchor="right",         # Ancrage à droite
                    x=0.99,                  # Position près de la droite
                    bgcolor="rgba(0,0,0,0.1)",  # Fond légèrement transparent
                    font=dict(size=9),
                    itemsizing='constant'  # Taille constante des items de légende
                ),
                margin=dict(l=50, r=20, t=20, b=70),  # Marges ajustées pour laisser plus d'espace en bas
                hoverlabel=dict(
                    bgcolor='#132158',  # Fond de l'infobulle assorti au thème
                    bordercolor='#0072ff',  # Bordure de l'infobulle
                    font=dict(color='white', size=10)  # Texte de l'infobulle
                )
            )
            
            # Créer les stats de corrélation
            correlation_class = "positive" if correlation_stats["correlation"] >= 0 else "negative"
            
            stats_div = [
                html.Div(
                    children=[
                        html.I(className=f"fas {'fa-arrow-up' if correlation_stats['correlation'] >= 0 else 'fa-arrow-down'} correlation-icon"),
                        html.Span(f"Corrélation: ", className="stat-label"),
                        html.Span(f"{correlation_stats['correlation']:.3f}", className=f"correlation {correlation_class}")
                    ],
                    className="stat-item"
                ),
                html.Div(
                    children=[
                        html.I(className="fas fa-ruler-combined slope-icon"),
                        html.Span(f"Pente: {correlation_stats['slope']:.3f}")
                    ],
                    className="stat-item"
                )
            ]
            
            return fig, stats_div
            
        except Exception as e:
            # En cas d'erreur, afficher une figure vide avec un message
            print(f"Erreur dans update_scatter_plot: {str(e)}")
            return empty_fig, error_stats
    
    # Callback pour mettre à jour les chiffres clés
    @app.callback(
        [
            Output('global-life-expectancy', 'children'),
            Output('global-population', 'children'),
            Output('growth-rate', 'children')
        ],
        [
            Input('year-slider', 'value')
        ]
    )
    def update_key_figures(year_range):
        """
        Met à jour les chiffres clés de l'application.
        
        Args:
            year_range (list): Liste [année_min, année_max]
            
        Returns:
            tuple: Triplet de valeurs mises à jour pour les chiffres clés
        """
        # Filtrer les données par plage d'années
        filtered_data = filter_data_by_year_range(data, year_range[0], year_range[1])
        
        # Au lieu de calculer sur toute la plage, prendre la dernière année
        last_year = year_range[1]
        last_year_data = filtered_data[filtered_data['Year'] == last_year]
        
        # Si aucune donnée n'est disponible pour la dernière année, prendre l'année la plus récente disponible
        if last_year_data.empty:
            available_years = filtered_data['Year'].unique()
            if len(available_years) > 0:
                last_year = max(available_years)
                last_year_data = filtered_data[filtered_data['Year'] == last_year]
            else:
                # Si toujours aucune donnée, utiliser toutes les données filtrées
                last_year_data = filtered_data
        
        # Calcul des métriques globales pour la dernière année
        total_population = last_year_data['Population, total'].sum()
        
        # Moyenne pondérée de l'espérance de vie
        life_exp_weighted = np.average(
            last_year_data['Life expectancy at birth, total (years)'], 
            weights=last_year_data['Population, total']
        )
        
        # Calcul du taux de croissance (si possible)
        try:
            # Obtenir les données de l'année précédente
            prev_year = last_year - 1
            prev_year_data = filtered_data[filtered_data['Year'] == prev_year]
            if not prev_year_data.empty:
                prev_total_population = prev_year_data['Population, total'].sum()
                growth_rate = ((total_population / prev_total_population) - 1) * 100
            else:
                # Si pas de données pour l'année précédente, utiliser données sur 5 ans ou une estimation
                growth_rate = 1.1  # Valeur par défaut
        except Exception as e:
            print(f"Erreur dans le calcul du taux de croissance: {e}")
            growth_rate = 1.1  # Valeur par défaut en cas d'erreur
        
        # Formater les nombres pour l'affichage
        formatted_life_exp = f"{life_exp_weighted:.1f} ans"
        # Utiliser un format qui évite la notation scientifique pour la population
        population_in_billions = total_population / 1e9
        formatted_population = f"{population_in_billions:.3f} Mrd"
        formatted_growth_rate = f"{growth_rate:.1f}%"
        
        return formatted_life_exp, formatted_population, formatted_growth_rate
    
    # Callback pour gérer les changements d'onglets
    @app.callback(
        Output('tabs', 'className'),
        [Input('tabs', 'value')]
    )
    def update_tab_style(selected_tab):
        """
        Met à jour le style des onglets en fonction de l'onglet sélectionné.
        
        Args:
            selected_tab (str): Identifiant de l'onglet sélectionné
            
        Returns:
            str: Classe CSS pour les onglets
        """
        if selected_tab == 'tab-data':
            return 'custom-tabs data-active'
        else:
            return 'custom-tabs' 