# World Population Dashboard

A professional dark blue data dashboard visualizing world population trends and forecasts based on World Bank data from 1960-2023.

## Overview

This dashboard provides an interactive visualization of global population data analysis, including:

- Historical population trends and forecasts to 2100
- Population distribution by continent
- Key demographic indicators (birth rate, life expectancy, population growth)
- Annual population growth rate over time

The dashboard is built using Dash and Plotly, with a dark blue professional theme.

## Data Sources

The dashboard uses data from:
- World Bank Development Indicators (1960-2023)
- Data processed in the Population.ipynb notebook

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

1. Navigate to the app directory:

```bash
cd app
```

2. Run the Dash application:

```bash
python app.py
```

3. Open your web browser and go to:

```
http://127.0.0.1:8050/
```

## Dashboard Sections

1. **Population Trend & Forecast** - Line chart showing historical population data and logistic growth forecast to 2100
2. **Population Distribution by Continent** - Bar chart showing population distribution across continents
3. **Global Demographic Indicators** - Key metrics with gauge visualizations
4. **Population Growth Rate Over Time** - Line chart showing annual population growth rate trends

## Features

- Professional dark blue theme (#0a0e2a) with blue/green visualization elements
- Responsive design that works on different screen sizes
- Interactive charts with hover information
- Clean, minimalist design with proper spacing and readability

## Development

The dashboard uses:
- Dash/Plotly for the web application and visualizations
- Pandas for data manipulation
- SciPy for curve fitting (logistic growth model)
- Custom CSS for styling 