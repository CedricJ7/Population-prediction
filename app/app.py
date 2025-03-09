import dash
from dash import dcc, html, Input, Output, State, callback
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
try:
    df_world = pd.read_csv("data/data_world.csv")
    df_countries = pd.read_csv("data/data_countries_imputed_iterative.csv")
except Exception as e:
    print(f"Error loading data: {e}")
    # Create empty dataframes for graceful fallback
    df_world = pd.DataFrame()
    df_countries = pd.DataFrame()

# Process data for the dashboard
def process_data():
    try:
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
        
        # Process world data
        world_pop_data = world_pop[["Year", "Population, total"]].copy()
        world_pop_data = world_pop_data.rename(columns={"Population, total": "Population"})
        world_pop_data["Population in Billions"] = world_pop_data["Population"] / 1e9
        
        # Get available years for controls
        available_years = sorted(world_pop["Year"].unique().tolist())
        
        # Process country data
        df_countries_processed = df_countries.copy()
        
        # Calculate growth rates
        world_pop_data["Growth Rate"] = world_pop_data["Population"].pct_change() * 100
        
        # Convert to numeric and handle missing values
        for col in numeric_cols:
            if col in df_countries_processed.columns:
                df_countries_processed[col] = pd.to_numeric(df_countries_processed[col], errors='coerce')
        
        # Add population in billions for countries
        df_countries_processed["Population in Billions"] = df_countries_processed["Population, total"] / 1e9
        
        # Get list of countries and continents for controls
        countries_list = sorted(df_countries_processed["Country Name"].unique().tolist())
        continents_list = sorted(df_countries_processed["Continent"].unique().tolist())
        
        # Get list of available metrics for controls
        metrics_list = [col for col in numeric_cols if col in df_countries_processed.columns]
        
        # Get the latest year for default display
        latest_year = max(available_years)
        
        return {
            "world_pop_data": world_pop_data,
            "countries_data": df_countries_processed,
            "available_years": available_years,
            "countries_list": countries_list,
            "continents_list": continents_list,
            "metrics_list": metrics_list,
            "latest_year": latest_year
        }
    except Exception as e:
        print(f"Error processing data: {e}")
        return {
            "world_pop_data": pd.DataFrame(),
            "countries_data": pd.DataFrame(),
            "available_years": [],
            "countries_list": [],
            "continents_list": [],
            "metrics_list": [],
            "latest_year": None
        }

# Process data
data = process_data()

# Continent color mapping for consistency
continent_colors = {
    'Africa': '#00a8ff',
    'Asia': '#0088ff',
    'Europe': '#26e282', 
    'North America': '#00ff88',
    'Oceania': '#00ccff',
    'South America': '#96c8ff'
}

# Define the app layout
app.layout = html.Div(
    style={
        'backgroundColor': '#0a0e2a',
        'color': 'white',
        'fontFamily': 'Arial, sans-serif',
        'minHeight': '100vh',
        'display': 'flex',
        'flexDirection': 'column'
    },
    children=[
        # Header
        html.Div(
            style={
                'padding': '20px 30px',
                'backgroundColor': '#090c24',
                'borderBottom': '1px solid #1e2542'
            },
            children=[
                html.H1("World Population Dashboard", 
                        style={'textAlign': 'center', 'color': 'white', 'margin': '0'}),
                html.P("Interactive analysis of global population trends based on World Bank data (1960-2023)",
                      style={'textAlign': 'center', 'color': '#00a8ff', 'marginTop': '10px', 'marginBottom': '0'})
            ]
        ),
        
        # Main Content Area
        html.Div(
            style={
                'display': 'flex',
                'flex': '1',
                'flexDirection': 'row'
            },
            children=[
                # Left Control Panel
                html.Div(
                    id="control-panel",
                    style={
                        'width': '250px',
                        'backgroundColor': '#121638',
                        'padding': '20px',
                        'borderRight': '1px solid #1e2542',
                        'overflowY': 'auto'
                    },
                    children=[
                        html.H3("Controls", style={'marginTop': '0', 'marginBottom': '20px', 'color': '#00a8ff'}),
                        
                        # Year Selection
                        html.Div(
                            style={'marginBottom': '25px'},
                            children=[
                                html.Label("Select Year", style={'marginBottom': '8px', 'display': 'block'}),
                                dcc.Slider(
                                    id='year-slider',
                                    min=min(data["available_years"]) if data["available_years"] else 1960,
                                    max=max(data["available_years"]) if data["available_years"] else 2023,
                                    step=1,
                                    marks={year: str(year) for year in 
                                          [min(data["available_years"]), max(data["available_years"])] 
                                          if data["available_years"] else {1960: '1960', 2023: '2023'}},
                                    value=data["latest_year"] if data["latest_year"] else 2023,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ]
                        ),
                        
                        # Continent Selection
                        html.Div(
                            style={'marginBottom': '25px'},
                            children=[
                                html.Label("Select Continents", style={'marginBottom': '8px', 'display': 'block'}),
                                dcc.Checklist(
                                    id='continent-checklist',
                                    options=[{'label': continent, 'value': continent} 
                                            for continent in data["continents_list"]],
                                    value=data["continents_list"],
                                    labelStyle={'display': 'block', 'marginBottom': '8px'},
                                    style={'maxHeight': '200px', 'overflowY': 'auto'}
                                )
                            ]
                        ),
                        
                        # Metrics Selection
                        html.Div(
                            style={'marginBottom': '25px'},
                            children=[
                                html.Label("Select Metric", style={'marginBottom': '8px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id='metric-dropdown',
                                    options=[{'label': metric.replace(', ', ' ').replace(' (', '\n('), 'value': metric} 
                                            for metric in data["metrics_list"]],
                                    value="Population, total" if "Population, total" in data["metrics_list"] else None,
                                    style={
                                        'backgroundColor': '#1e2542',
                                        'color': 'white',
                                        'border': 'none'
                                    }
                                )
                            ]
                        ),
                        
                        # Number of Countries to Show
                        html.Div(
                            style={'marginBottom': '25px'},
                            children=[
                                html.Label("Top Countries to Show", style={'marginBottom': '8px', 'display': 'block'}),
                                dcc.Slider(
                                    id='top-n-slider',
                                    min=5,
                                    max=20,
                                    step=5,
                                    marks={i: str(i) for i in range(5, 21, 5)},
                                    value=10,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ]
                        ),
                        
                        # Forecast Toggle
                        html.Div(
                            style={'marginBottom': '25px'},
                            children=[
                                html.Label("Show Forecast", style={'marginBottom': '8px', 'display': 'block'}),
                                dcc.RadioItems(
                                    id='forecast-toggle',
                                    options=[
                                        {'label': 'Yes', 'value': 'yes'},
                                        {'label': 'No', 'value': 'no'}
                                    ],
                                    value='yes',
                                    inline=True
                                )
                            ]
                        ),
                        
                        # Apply Button
                        html.Button(
                            "Apply Filters",
                            id="apply-button",
                            style={
                                'backgroundColor': '#00a8ff',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 15px',
                                'borderRadius': '5px',
                                'cursor': 'pointer',
                                'width': '100%',
                                'marginTop': '10px',
                                'fontWeight': 'bold'
                            }
                        ),
                        
                        # Reset Button
                        html.Button(
                            "Reset Filters",
                            id="reset-button",
                            style={
                                'backgroundColor': 'transparent',
                                'color': 'white',
                                'border': '1px solid #00a8ff',
                                'padding': '8px 15px',
                                'borderRadius': '5px',
                                'cursor': 'pointer',
                                'width': '100%',
                                'marginTop': '10px'
                            }
                        )
                    ]
                ),
                
                # Main Content Area
                html.Div(
                    id="main-content",
                    style={
                        'flex': '1',
                        'padding': '20px',
                        'overflowY': 'auto'
                    },
                    children=[
                        # First row: Key Metrics and World Population Trend
                        html.Div(
                            style={
                                'display': 'flex',
                                'marginBottom': '20px',
                                'gap': '20px'
                            },
                            children=[
                                # Population Evolution Card
                                html.Div(
                                    style={
                                        'flex': '1',
                                        'backgroundColor': '#121638',
                                        'borderRadius': '10px',
                                        'padding': '20px',
                                        'minHeight': '400px'
                                    },
                                    children=[
                                        html.H3("Global Population Trend & Forecast", 
                                              style={'color': 'white', 'marginTop': '0', 'marginBottom': '15px'}),
                                        dcc.Graph(
                                            id='global-population-trend',
                                            style={'height': '350px'},
                                            config={'displayModeBar': False}
                                        )
                                    ]
                                )
                            ]
                        ),
                        
                        # Second row: Demographics & Map
                        html.Div(
                            style={
                                'display': 'flex',
                                'marginBottom': '20px',
                                'gap': '20px'
                            },
                            children=[
                                # Demographic Indicators
                                html.Div(
                                    style={
                                        'flex': '1',
                                        'backgroundColor': '#121638',
                                        'borderRadius': '10px',
                                        'padding': '20px'
                                    },
                                    children=[
                                        html.H3("Population Distribution by Continent", 
                                              style={'color': 'white', 'marginTop': '0', 'marginBottom': '15px'}),
                                        dcc.Graph(
                                            id='continent-distribution',
                                            style={'height': '350px'},
                                            config={'displayModeBar': False}
                                        )
                                    ]
                                ),
                                
                                # Map Visualization
                                html.Div(
                                    style={
                                        'flex': '1',
                                        'backgroundColor': '#121638',
                                        'borderRadius': '10px',
                                        'padding': '20px'
                                    },
                                    children=[
                                        html.H3("Geographic Population Distribution", 
                                              style={'color': 'white', 'marginTop': '0', 'marginBottom': '15px'}),
                                        dcc.Graph(
                                            id='population-map',
                                            style={'height': '350px'},
                                            config={'displayModeBar': False}
                                        )
                                    ]
                                )
                            ]
                        ),
                        
                        # Third row: Top Countries & Birth/Death Rates
                        html.Div(
                            style={
                                'display': 'flex',
                                'marginBottom': '20px',
                                'gap': '20px'
                            },
                            children=[
                                # Top Countries Chart
                                html.Div(
                                    style={
                                        'flex': '1',
                                        'backgroundColor': '#121638',
                                        'borderRadius': '10px',
                                        'padding': '20px'
                                    },
                                    children=[
                                        html.H3("Top Countries by Selected Metric", 
                                              style={'color': 'white', 'marginTop': '0', 'marginBottom': '15px'}),
                                        dcc.Graph(
                                            id='top-countries',
                                            style={'height': '350px'},
                                            config={'displayModeBar': False}
                                        )
                                    ]
                                ),
                                
                                # Birth & Death Rates
                                html.Div(
                                    style={
                                        'flex': '1',
                                        'backgroundColor': '#121638',
                                        'borderRadius': '10px',
                                        'padding': '20px'
                                    },
                                    children=[
                                        html.H3("Birth and Death Rates Over Time", 
                                              style={'color': 'white', 'marginTop': '0', 'marginBottom': '15px'}),
                                        dcc.Graph(
                                            id='vital-rates',
                                            style={'height': '350px'},
                                            config={'displayModeBar': False}
                                        )
                                    ]
                                )
                            ]
                        ),
                        
                        # Fourth row: Life Expectancy & Population Growth
                        html.Div(
                            style={
                                'display': 'flex',
                                'marginBottom': '20px',
                                'gap': '20px'
                            },
                            children=[
                                # Life Expectancy by Continent
                                html.Div(
                                    style={
                                        'flex': '1',
                                        'backgroundColor': '#121638',
                                        'borderRadius': '10px',
                                        'padding': '20px'
                                    },
                                    children=[
                                        html.H3("Life Expectancy by Continent", 
                                              style={'color': 'white', 'marginTop': '0', 'marginBottom': '15px'}),
                                        dcc.Graph(
                                            id='life-expectancy',
                                            style={'height': '350px'},
                                            config={'displayModeBar': False}
                                        )
                                    ]
                                ),
                                
                                # Growth Rate Timeline
                                html.Div(
                                    style={
                                        'flex': '1',
                                        'backgroundColor': '#121638',
                                        'borderRadius': '10px',
                                        'padding': '20px'
                                    },
                                    children=[
                                        html.H3("Population Growth Rate", 
                                              style={'color': 'white', 'marginTop': '0', 'marginBottom': '15px'}),
                                        dcc.Graph(
                                            id='growth-rate',
                                            style={'height': '350px'},
                                            config={'displayModeBar': False}
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        
        # Footer
        html.Div(
            style={
                'padding': '15px',
                'textAlign': 'center',
                'borderTop': '1px solid #1e2542',
                'backgroundColor': '#090c24'
            },
            children=[
                html.P("Data source: World Bank Development Indicators (1960-2023)",
                      style={'margin': '0', 'color': '#8a8d98'})
            ]
        )
    ]
)

# Callbacks
@app.callback(
    [
        Output('global-population-trend', 'figure'),
        Output('continent-distribution', 'figure'),
        Output('population-map', 'figure'),
        Output('top-countries', 'figure'),
        Output('vital-rates', 'figure'),
        Output('life-expectancy', 'figure'),
        Output('growth-rate', 'figure')
    ],
    [
        Input('apply-button', 'n_clicks')
    ],
    [
        State('year-slider', 'value'),
        State('continent-checklist', 'value'),
        State('metric-dropdown', 'value'),
        State('top-n-slider', 'value'),
        State('forecast-toggle', 'value')
    ],
    prevent_initial_call=False
)
def update_dashboard(n_clicks, selected_year, selected_continents, selected_metric, top_n, show_forecast):
    # Set up context
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'apply-button'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Get data
    world_pop_data = data["world_pop_data"]
    countries_data = data["countries_data"]
    
    # Handle empty data case
    if world_pop_data.empty or countries_data.empty:
        empty_fig = px.scatter().update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"),
            xaxis=dict(gridcolor="#1e2542"),
            yaxis=dict(gridcolor="#1e2542"),
            title="No data available"
        )
        return [empty_fig] * 7
    
    # Filter data by selected continents
    if selected_continents:
        filtered_countries = countries_data[countries_data['Continent'].isin(selected_continents)]
    else:
        filtered_countries = countries_data
    
    # Use default metric if none selected
    if not selected_metric:
        selected_metric = "Population, total"
    
    # 1. Global Population Trend & Forecast
    def generate_population_forecast(years_hist, pop_hist, forecast_years=None):
        def logistic_model(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))
        
        # Normalize years for numerical stability
        year_min = years_hist.min()
        years_hist_norm = years_hist - year_min
        
        try:
            # Fit logistic model
            p0 = [11, 0.05, 70]  # Initial guess
            params, _ = curve_fit(logistic_model, years_hist_norm, pop_hist, p0=p0, maxfev=10000)
            L, k, x0 = params
            
            if forecast_years is not None:
                forecast_years_norm = forecast_years - year_min
                forecast_pop = logistic_model(forecast_years_norm, L, k, x0)
                return forecast_years, forecast_pop
            else:
                return None, None
        except:
            return None, None
    
    # Prepare population trend data
    pop_trend_fig = px.line(
        world_pop_data, 
        x="Year", 
        y="Population in Billions",
        title="Global Population Trend (1960-2023)",
        labels={"Population in Billions": "Population (billions)", "Year": "Year"}
    )
    
    pop_trend_fig.update_traces(line_color='#00a8ff', line_width=3)
    
    # Add forecast if requested
    if show_forecast == 'yes':
        years_hist = world_pop_data["Year"].values
        pop_hist = world_pop_data["Population in Billions"].values
        forecast_years = np.arange(max(years_hist) + 1, 2101)
        
        f_years, f_pop = generate_population_forecast(years_hist, pop_hist, forecast_years)
        
        if f_years is not None and f_pop is not None:
            pop_trend_fig.add_scatter(
                x=f_years, 
                y=f_pop, 
                mode='lines',
                name='Forecast',
                line=dict(color='#26e282', width=3, dash='dash')
            )
            pop_trend_fig.update_layout(title="Global Population Trend & Forecast to 2100")
    
    pop_trend_fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        xaxis=dict(gridcolor="#1e2542"),
        yaxis=dict(gridcolor="#1e2542"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
    )
    
    # 2. Continent Distribution
    # Filter by year
    year_data = filtered_countries[filtered_countries["Year"] == selected_year]
    continent_data = year_data.groupby("Continent")["Population in Billions"].sum().reset_index()
    
    # Sort by population in descending order
    continent_data = continent_data.sort_values("Population in Billions", ascending=False)
    
    continent_fig = px.bar(
        continent_data,
        x="Continent",
        y="Population in Billions",
        color="Continent",
        color_discrete_map=continent_colors,
        title=f"Population Distribution by Continent ({selected_year})",
        labels={"Population in Billions": "Population (billions)", "Continent": "Continent"}
    )
    
    continent_fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        xaxis=dict(gridcolor="#1e2542"),
        yaxis=dict(gridcolor="#1e2542"),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False
    )
    
    # 3. Map Visualization
    map_data = year_data.copy()
    
    # Use selected metric for the map (default to population)
    map_metric = selected_metric if selected_metric in map_data.columns else "Population, total"
    
    map_fig = px.choropleth(
        map_data,
        locations="Country Code",
        color=map_metric,
        hover_name="Country Name",
        color_continuous_scale=["#121638", "#00a8ff", "#26e282"],
        title=f"Geographic Distribution of {map_metric.split('(')[0]} ({selected_year})",
        projection="natural earth"
    )
    
    map_fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=30, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="equirectangular",
            bgcolor='rgba(0,0,0,0)'
        )
    )
    
    # 4. Top Countries Chart
    # Get top N countries by the selected metric
    if selected_metric in year_data.columns:
        top_countries = year_data.sort_values(by=selected_metric, ascending=False).head(top_n)
        
        # Create the bar chart sorted in descending order
        top_countries_fig = px.bar(
            top_countries,
            x="Country Name",
            y=selected_metric,
            color="Continent",
            color_discrete_map=continent_colors,
            title=f"Top {top_n} Countries by {selected_metric.split('(')[0]} ({selected_year})",
            labels={selected_metric: selected_metric.split('(')[0], "Country Name": "Country"}
        )
        
        top_countries_fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"),
            xaxis=dict(gridcolor="#1e2542", categoryorder='total descending'),
            yaxis=dict(gridcolor="#1e2542"),
            margin=dict(l=10, r=10, t=30, b=10)
        )
    else:
        # Fallback to population if metric not available
        top_countries_fig = px.bar(
            title="Selected metric not available in data"
        ).update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )
    
    # 5. Birth and Death Rates Over Time
    birth_death_data = world_pop_data.copy()
    
    # Create birth/death rates chart
    vital_rates_data = filtered_countries.groupby("Year")[
        ["Birth rate, crude (per 1,000 people)", "Death rate, crude (per 1,000 people)"]
    ].mean().reset_index()
    
    vital_rates_melted = pd.melt(
        vital_rates_data,
        id_vars=["Year"],
        value_vars=["Birth rate, crude (per 1,000 people)", "Death rate, crude (per 1,000 people)"],
        var_name="Rate Type",
        value_name="Rate"
    )
    
    vital_rates_melted["Rate Type"] = vital_rates_melted["Rate Type"].replace({
        "Birth rate, crude (per 1,000 people)": "Birth Rate",
        "Death rate, crude (per 1,000 people)": "Death Rate"
    })
    
    vital_rates_fig = px.line(
        vital_rates_melted,
        x="Year",
        y="Rate",
        color="Rate Type",
        title="Birth and Death Rates Over Time",
        labels={"Rate": "Rate per 1,000 people", "Year": "Year", "Rate Type": ""},
        color_discrete_map={"Birth Rate": "#00a8ff", "Death Rate": "#ff5555"}
    )
    
    vital_rates_fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        xaxis=dict(gridcolor="#1e2542"),
        yaxis=dict(gridcolor="#1e2542"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
    )
    
    # 6. Life Expectancy by Continent
    life_exp_data = filtered_countries[filtered_countries["Year"] == selected_year].groupby("Continent")[
        "Life expectancy at birth, total (years)"
    ].mean().reset_index()
    
    # Sort in descending order
    life_exp_data = life_exp_data.sort_values("Life expectancy at birth, total (years)", ascending=False)
    
    life_exp_fig = px.bar(
        life_exp_data,
        x="Continent",
        y="Life expectancy at birth, total (years)",
        color="Continent",
        color_discrete_map=continent_colors,
        title=f"Life Expectancy by Continent ({selected_year})",
        labels={
            "Life expectancy at birth, total (years)": "Life Expectancy (years)", 
            "Continent": "Continent"
        }
    )
    
    life_exp_fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        xaxis=dict(gridcolor="#1e2542"),
        yaxis=dict(gridcolor="#1e2542"),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False
    )
    
    # 7. Growth Rate Timeline
    growth_data = filtered_countries.groupby("Year")["Population growth (annual %)"].mean().reset_index()
    
    growth_fig = px.line(
        growth_data,
        x="Year",
        y="Population growth (annual %)",
        title="Population Growth Rate Over Time",
        labels={"Population growth (annual %)": "Growth Rate (%)", "Year": "Year"}
    )
    
    growth_fig.update_traces(line_color='#26e282', line_width=3)
    
    growth_fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        xaxis=dict(gridcolor="#1e2542"),
        yaxis=dict(gridcolor="#1e2542"),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    return [
        pop_trend_fig,
        continent_fig,
        map_fig,
        top_countries_fig,
        vital_rates_fig,
        life_exp_fig,
        growth_fig
    ]

# Reset button callback
@app.callback(
    [
        Output('year-slider', 'value'),
        Output('continent-checklist', 'value'),
        Output('metric-dropdown', 'value'),
        Output('top-n-slider', 'value'),
        Output('forecast-toggle', 'value')
    ],
    [Input('reset-button', 'n_clicks')],
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return [
        data["latest_year"],
        data["continents_list"],
        "Population, total" if "Population, total" in data["metrics_list"] else None,
        10,
        'yes'
    ]

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
