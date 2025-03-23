# Interactive World Population Dashboard

An interactive data dashboard visualizing world population trends and forecasts based on World Bank data (1960-2023), featuring a responsive layout with control panel and dynamic visualizations.

## Overview

This redesigned dashboard provides an interactive analysis of global population data with:

- Left control panel for filtering and customizing visualizations
- Dynamic updating of visualizations based on user selections
- Geographic distribution map
- Time series analysis of population trends
- Comprehensive demographic indicators
- Forecasting capabilities

## Data Sources

The dashboard uses data from:
- World Bank Development Indicators (1960-2023)
- Data processed from the population.ipynb notebook analysis

## Features

- **Interactive Controls**:
  - Year selector (1960-2023)
  - Continent filter
  - Metric selector
  - Top countries count adjustment
  - Forecast toggle
  
- **Visualizations**:
  - Population trend with optional forecast to 2100
  - Geographic choropleth map showing selected metrics
  - Continental population distribution
  - Top countries by selected metric
  - Birth and death rates over time
  - Life expectancy by continent
  - Population growth rate analysis

- **Design**:
  - Professional dark blue theme (#0a0e2a)
  - Blue/green visualization elements for optimal readability
  - Responsive layout that works on different screen sizes
  - Clean, minimalist aesthetic with proper spacing
  - Consistent color coding across visualizations

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

## Usage Instructions

1. **Filtering Data**: 
   - Use the left panel to select years, continents, and metrics
   - Click "Apply Filters" to update visualizations
   - Click "Reset Filters" to return to default settings

2. **Interacting with Visualizations**:
   - Hover over charts for detailed tooltips
   - Click on legend items to toggle visibility
   - Use the map controls to zoom and pan

3. **Analyzing Trends**:
   - Toggle forecast display on/off
   - Compare metrics across continents and countries
   - Examine trends over time with the time-series visualizations

## Technical Implementation

The dashboard uses:
- Dash/Plotly for the web application and visualizations
- Pandas for data manipulation
- SciPy for curve fitting (logistic growth model)
- CSS for responsive styling
- Callbacks for interactive filtering 