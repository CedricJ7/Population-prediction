import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Load data
df_world = pd.read_csv("data/data_world.csv")
df_countries = pd.read_csv("data/data_countries_imputed_iterative.csv")

# Process data for the dashboard
def process_data():
    # World population data
    world_pop = df_world.copy()
    
    # Replace '..' with NaN
    for col in world_pop.columns:
        if world_pop[col].dtype == object:
            world_pop[col] = world_pop[col].replace('..', np.nan)
    
    # Convert columns to numeric where possible
    numeric_cols = ["Population, total", "Birth rate, crude (per 1,000 people)", 
                   "Death rate, crude (per 1,000 people)", "Life expectancy at birth, total (years)",
                   "Population growth (annual %)", "GDP per capita (current US$)",
                   "Exports of goods and services (% of GDP)", "Electric power consumption (kWh per capita)",
                   "Adjusted net national income (current US$)"]
    for col in numeric_cols:
        if col in world_pop.columns:
            world_pop[col] = pd.to_numeric(world_pop[col], errors='coerce')
    
    world_pop_data = world_pop[["Year", "Population, total"]].copy()
    world_pop_data = world_pop_data.rename(columns={"Population, total": "Population"})
    world_pop_data["Population in Billions"] = world_pop_data["Population"] / 1e9
    
    # Demographic metrics
    latest_year = world_pop["Year"].max()
    latest_data = world_pop[world_pop["Year"] == latest_year].iloc[0]
    
    # Aggregate by continent
    df_countries["Population in Billions"] = df_countries["Population, total"] / 1e9
    continent_pop = df_countries.groupby(["Year", "Continent"])["Population in Billions"].sum().reset_index()
    current_continent_pop = continent_pop[continent_pop["Year"] == latest_year]
    
    # Population by continent over time
    continent_pop_time = df_countries.groupby(["Year", "Continent"])["Population in Billions"].sum().reset_index()
    
    # Correlation matrix data
    correlation_cols = ["Population, total", "Birth rate, crude (per 1,000 people)", 
                        "Death rate, crude (per 1,000 people)", "Life expectancy at birth, total (years)",
                        "Population growth (annual %)", "GDP per capita (current US$)"]
    corr_data = world_pop[correlation_cols].corr().reset_index()
    corr_data = corr_data.rename(columns={"index": "Indicator"})
    
    # Prepare data for top 10 most populous countries
    top_countries = df_countries[df_countries["Year"] == latest_year].sort_values(
        by="Population, total", ascending=False).head(10)
    
    # GDP vs Population data
    gdp_pop_data = df_countries[df_countries["Year"] == latest_year].copy()
    gdp_pop_data = gdp_pop_data.dropna(subset=["GDP per capita (current US$)", "Population, total"])
    
    # Life expectancy by continent
    life_exp_continent = df_countries.groupby(["Year", "Continent"])["Life expectancy at birth, total (years)"].mean().reset_index()
    current_life_exp = life_exp_continent[life_exp_continent["Year"] == latest_year]
    
    # Forecast world population with logistic growth model
    years_hist = world_pop_data["Year"].values
    pop_hist = world_pop_data["Population in Billions"].values
    forecast_years = np.arange(latest_year + 1, 2101)
    
    def logistic_model(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    # Normalize years for numerical stability
    year_min = years_hist.min()
    years_hist_norm = years_hist - year_min
    forecast_years_norm = forecast_years - year_min
    
    # Fit logistic model
    try:
        p0 = [11, 0.05, 70]  # Initial guess
        params, _ = curve_fit(logistic_model, years_hist_norm, pop_hist, p0=p0, maxfev=10000)
        L, k, x0 = params
        forecast_pop = logistic_model(forecast_years_norm, L, k, x0)
        
        # Combine historical and forecast
        combined_years = np.concatenate([years_hist, forecast_years])
        combined_pop = np.concatenate([pop_hist, forecast_pop])
        forecast_df = pd.DataFrame({
            "Year": combined_years,
            "Population in Billions": combined_pop,
            "Type": ["Historical" if y <= latest_year else "Forecast" for y in combined_years]
        })
    except:
        # Fallback if curve fitting fails
        forecast_df = pd.DataFrame({
            "Year": years_hist,
            "Population in Billions": pop_hist,
            "Type": ["Historical" for _ in years_hist]
        })
    
    # Demographic indicators for latest year with default values
    demographic_data = {
        "Birth Rate": latest_data.get("Birth rate, crude (per 1,000 people)", 0),
        "Death Rate": latest_data.get("Death rate, crude (per 1,000 people)", 0),
        "Life Expectancy": latest_data.get("Life expectancy at birth, total (years)", 0),
        "Population Growth": latest_data.get("Population growth (annual %)", 0)
    }
    
    # Ensure all values are numeric
    for key in demographic_data:
        if pd.isna(demographic_data[key]) or demographic_data[key] == "..":
            demographic_data[key] = 0
    
    # Calculate growth rates for visualization
    world_pop_data["Growth Rate"] = world_pop_data["Population"].pct_change() * 100
    
    # Birth and death rates over time
    vital_rates = world_pop[["Year", "Birth rate, crude (per 1,000 people)", "Death rate, crude (per 1,000 people)"]].copy()
    vital_rates = vital_rates.rename(columns={
        "Birth rate, crude (per 1,000 people)": "Birth Rate",
        "Death rate, crude (per 1,000 people)": "Death Rate"
    })
    vital_rates = vital_rates.melt(id_vars=["Year"], var_name="Rate Type", value_name="Rate")
    
    return (world_pop_data, forecast_df, demographic_data, current_continent_pop, 
            corr_data, top_countries, gdp_pop_data, life_exp_continent, 
            continent_pop_time, vital_rates)

(world_pop_data, forecast_df, demographic_data, continent_pop, 
 corr_data, top_countries, gdp_pop_data, life_exp_continent, 
 continent_pop_time, vital_rates) = process_data()

# Define the app layout
app.layout = html.Div(
    style={
        'backgroundColor': '#0a0e2a',
        'color': 'white',
        'fontFamily': 'Arial, sans-serif',
        'padding': '20px',
        'minHeight': '100vh'
    },
    children=[
        # Header
        html.Div([
            html.H1("World Population Dashboard", 
                    style={'textAlign': 'center', 'color': 'white', 'marginBottom': '20px'}),
            html.P("Analysis of global population trends and forecasts based on World Bank data (1960-2023)",
                  style={'textAlign': 'center', 'color': '#00a8ff', 'marginBottom': '30px'})
        ]),
        
        # First row: Population Timeline & Forecast
        html.Div([
            html.Div([
                html.H3("Global Population Trend & Forecast",
                       style={'color': 'white', 'marginBottom': '15px'}),
                dcc.Graph(
                    id='population-timeline',
                    figure=px.line(
                        forecast_df, 
                        x="Year", 
                        y="Population in Billions",
                        color="Type",
                        color_discrete_map={"Historical": "#00a8ff", "Forecast": "#26e282"},
                        title="Population Growth & Forecast to 2100"
                    ).update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(gridcolor="#1e2542"),
                        yaxis=dict(gridcolor="#1e2542")
                    )
                )
            ], style={'width': '100%', 'backgroundColor': '#121638', 'padding': '20px', 'borderRadius': '10px'})
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        # Second row: Continental Distribution & Key Metrics
        html.Div([
            # Continental Distribution
            html.Div([
                html.H3("Population Distribution by Continent",
                       style={'color': 'white', 'marginBottom': '15px'}),
                dcc.Graph(
                    id='continent-distribution',
                    figure=px.bar(
                        continent_pop,
                        x="Continent",
                        y="Population in Billions",
                        color="Continent",
                        color_discrete_sequence=['#00a8ff', '#0088ff', '#26e282', '#00ff88', '#00ccff', '#96c8ff'],
                        title="Current Population by Continent"
                    ).update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        xaxis=dict(gridcolor="#1e2542"),
                        yaxis=dict(gridcolor="#1e2542")
                    )
                )
            ], style={'width': '60%', 'backgroundColor': '#121638', 'padding': '20px', 'borderRadius': '10px', 'marginRight': '20px'}),
            
            # Key Metrics
            html.Div([
                html.H3("Global Demographic Indicators",
                       style={'color': 'white', 'marginBottom': '15px'}),
                html.Div([
                    html.Div([
                        html.H4("Birth Rate", style={'color': 'white', 'marginBottom': '10px', 'textAlign': 'center'}),
                        html.Div([
                            dcc.Graph(
                                id='birth-rate-indicator',
                                figure=go.Figure(
                                    go.Indicator(
                                        mode="gauge+number",
                                        value=float(demographic_data["Birth Rate"]),
                                        title={'text': "per 1,000 people", 'font': {'size': 14, 'color': 'white'}},
                                        gauge={'axis': {'range': [0, 50], 'tickwidth': 1, 'tickcolor': "white"},
                                               'bar': {'color': "#00a8ff"},
                                               'bgcolor': "rgba(0,0,0,0)",
                                               'borderwidth': 2,
                                               'bordercolor': "#1e2542",
                                               'steps': [
                                                   {'range': [0, 15], 'color': 'rgba(0, 168, 255, 0.3)'},
                                                   {'range': [15, 30], 'color': 'rgba(0, 168, 255, 0.6)'},
                                                   {'range': [30, 50], 'color': 'rgba(0, 168, 255, 0.9)'}
                                               ]
                                              }
                                    )
                                ).update_layout(
                                    height=150,
                                    template="plotly_dark",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color="white"),
                                    margin=dict(l=20, r=20, t=30, b=20)
                                )
                            )
                        ]),
                    ], style={'marginBottom': '15px'}),
                    
                    html.Div([
                        html.H4("Life Expectancy", style={'color': 'white', 'marginBottom': '10px', 'textAlign': 'center'}),
                        dcc.Graph(
                            id='life-expectancy-indicator',
                            figure=go.Figure(
                                go.Indicator(
                                    mode="gauge+number",
                                    value=float(demographic_data["Life Expectancy"]),
                                    title={'text': "years", 'font': {'size': 14, 'color': 'white'}},
                                    gauge={'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                           'bar': {'color': "#26e282"},
                                           'bgcolor': "rgba(0,0,0,0)",
                                           'borderwidth': 2,
                                           'bordercolor': "#1e2542",
                                           'steps': [
                                               {'range': [0, 50], 'color': 'rgba(38, 226, 130, 0.3)'},
                                               {'range': [50, 75], 'color': 'rgba(38, 226, 130, 0.6)'},
                                               {'range': [75, 100], 'color': 'rgba(38, 226, 130, 0.9)'}
                                           ]
                                          }
                                )
                            ).update_layout(
                                height=150,
                                template="plotly_dark",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color="white"),
                                margin=dict(l=20, r=20, t=30, b=20)
                            )
                        )
                    ], style={'marginBottom': '15px'}),
                    
                    html.Div([
                        html.H4("Population Growth", style={'color': 'white', 'marginBottom': '10px', 'textAlign': 'center'}),
                        dcc.Graph(
                            id='population-growth-indicator',
                            figure=go.Figure(
                                go.Indicator(
                                    mode="gauge+number",
                                    value=float(demographic_data["Population Growth"]),
                                    number={'suffix': '%'},
                                    title={'text': "annual growth", 'font': {'size': 14, 'color': 'white'}},
                                    gauge={'axis': {'range': [0, 3], 'tickwidth': 1, 'tickcolor': "white"},
                                           'bar': {'color': "#00ccff"},
                                           'bgcolor': "rgba(0,0,0,0)",
                                           'borderwidth': 2,
                                           'bordercolor': "#1e2542",
                                           'steps': [
                                               {'range': [0, 1], 'color': 'rgba(0, 204, 255, 0.3)'},
                                               {'range': [1, 2], 'color': 'rgba(0, 204, 255, 0.6)'},
                                               {'range': [2, 3], 'color': 'rgba(0, 204, 255, 0.9)'}
                                           ]
                                          }
                                )
                            ).update_layout(
                                height=150,
                                template="plotly_dark",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color="white"),
                                margin=dict(l=20, r=20, t=30, b=20)
                            )
                        )
                    ])
                ])
            ], style={'width': '40%', 'backgroundColor': '#121638', 'padding': '20px', 'borderRadius': '10px'})
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        # Third row: Population Growth Rate Timeline
        html.Div([
            html.Div([
                html.H3("Population Growth Rate Over Time",
                       style={'color': 'white', 'marginBottom': '15px'}),
                dcc.Graph(
                    id='growth-rate-timeline',
                    figure=px.line(
                        world_pop_data.dropna(subset=["Growth Rate"]), 
                        x="Year", 
                        y="Growth Rate",
                        title="Annual Population Growth Rate (%)",
                        color_discrete_sequence=["#26e282"]
                    ).update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        xaxis=dict(gridcolor="#1e2542"),
                        yaxis=dict(gridcolor="#1e2542", title="Growth Rate (%)")
                    )
                )
            ], style={'width': '100%', 'backgroundColor': '#121638', 'padding': '20px', 'borderRadius': '10px'})
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        # Fourth row: New Descriptive Analysis - Birth and Death Rates
        html.Div([
            html.Div([
                html.H3("Birth and Death Rates Over Time",
                       style={'color': 'white', 'marginBottom': '15px'}),
                dcc.Graph(
                    id='vital-rates-timeline',
                    figure=px.line(
                        vital_rates.dropna(), 
                        x="Year", 
                        y="Rate",
                        color="Rate Type",
                        title="Birth and Death Rates per 1,000 People",
                        color_discrete_map={"Birth Rate": "#00a8ff", "Death Rate": "#ff5555"}
                    ).update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        xaxis=dict(gridcolor="#1e2542"),
                        yaxis=dict(gridcolor="#1e2542")
                    )
                )
            ], style={'width': '100%', 'backgroundColor': '#121638', 'padding': '20px', 'borderRadius': '10px'})
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        # Fifth row: Correlation Matrix and Top 10 Countries
        html.Div([
            # Correlation Matrix
            html.Div([
                html.H3("Correlation Matrix of Key Indicators",
                       style={'color': 'white', 'marginBottom': '15px'}),
                dcc.Graph(
                    id='correlation-matrix',
                    figure=px.imshow(
                        corr_data.set_index('Indicator'),
                        color_continuous_scale=["#0a0e2a", "#121638", "#00a8ff", "#26e282"],
                        title="Correlation Between Key Demographic Indicators"
                    ).update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white")
                    )
                )
            ], style={'width': '60%', 'backgroundColor': '#121638', 'padding': '20px', 'borderRadius': '10px', 'marginRight': '20px'}),
            
            # Top 10 Most Populous Countries
            html.Div([
                html.H3("Top 10 Most Populous Countries",
                       style={'color': 'white', 'marginBottom': '15px'}),
                dcc.Graph(
                    id='top-countries',
                    figure=px.bar(
                        top_countries,
                        x="Country Name",
                        y="Population, total",
                        color="Continent",
                        title="Top 10 Countries by Population",
                        color_discrete_sequence=['#00a8ff', '#0088ff', '#26e282', '#00ff88', '#00ccff', '#96c8ff']
                    ).update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        xaxis=dict(gridcolor="#1e2542", categoryorder='total descending'),
                        yaxis=dict(gridcolor="#1e2542", title="Population")
                    )
                )
            ], style={'width': '40%', 'backgroundColor': '#121638', 'padding': '20px', 'borderRadius': '10px'})
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        # Sixth row: Population by Continent Over Time and Life Expectancy by Continent
        html.Div([
            # Population by Continent Over Time
            html.Div([
                html.H3("Population by Continent Over Time",
                       style={'color': 'white', 'marginBottom': '15px'}),
                dcc.Graph(
                    id='continent-time-series',
                    figure=px.line(
                        continent_pop_time,
                        x="Year",
                        y="Population in Billions",
                        color="Continent",
                        title="Evolution of Population by Continent",
                        color_discrete_sequence=['#00a8ff', '#0088ff', '#26e282', '#00ff88', '#00ccff', '#96c8ff']
                    ).update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        xaxis=dict(gridcolor="#1e2542"),
                        yaxis=dict(gridcolor="#1e2542")
                    )
                )
            ], style={'width': '50%', 'backgroundColor': '#121638', 'padding': '20px', 'borderRadius': '10px', 'marginRight': '20px'}),
            
            # Life Expectancy by Continent
            html.Div([
                html.H3("Life Expectancy by Continent",
                       style={'color': 'white', 'marginBottom': '15px'}),
                dcc.Graph(
                    id='life-expectancy-continent',
                    figure=px.bar(
                        life_exp_continent[life_exp_continent["Year"] == life_exp_continent["Year"].max()],
                        x="Continent",
                        y="Life expectancy at birth, total (years)",
                        color="Continent",
                        title="Current Life Expectancy by Continent",
                        color_discrete_sequence=['#00a8ff', '#0088ff', '#26e282', '#00ff88', '#00ccff', '#96c8ff']
                    ).update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        xaxis=dict(gridcolor="#1e2542"),
                        yaxis=dict(gridcolor="#1e2542", title="Life Expectancy (years)")
                    )
                )
            ], style={'width': '50%', 'backgroundColor': '#121638', 'padding': '20px', 'borderRadius': '10px'})
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        # Footer
        html.Div([
            html.P("Data source: World Bank Development Indicators (1960-2023)",
                  style={'textAlign': 'center', 'color': '#8a8d98', 'marginTop': '20px'})
        ])
    ]
)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
