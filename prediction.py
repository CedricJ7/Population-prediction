import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Pour ARIMAX (ARIMA avec variables exogènes)
from statsmodels.tsa.arima.model import ARIMA
# Pour RandomForest, Ridge et XGBoost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
# Pour LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def compare_models_and_plot():
    # --- Chargement des données ---
    data = pd.read_csv("data/data_countries_imputed_iterative.csv")
    print("Colonnes du DataFrame :", data.columns.tolist())
    
    target = "Population, total"
    predictive_cols = [
        'Year',
        'Adjusted net national income (current US$)',
        'Birth rate, crude (per 1,000 people)',
        'Death rate, crude (per 1,000 people)',
        'Exports of goods and services (% of GDP)',
        'GDP per capita (current US$)',
        'Life expectancy at birth, total (years)',
        'Population growth (annual %)',
        'Country Name',
        'Continent'
    ]
    
    # --- Définition des périodes ---
    train_data = data[data["Year"] < 2016].copy()
    test_data = data[(data["Year"] >= 2016) & (data["Year"] <= 2023)].copy()
    full_data = data[data["Year"] <= 2023].copy()  # pour la prévision
    forecast_years = np.arange(2016, 2051)
    
    # --- Préparation des données panel (pour RF, XGBoost, Ridge, LSTM) ---
    X_train = pd.get_dummies(train_data[predictive_cols], drop_first=True)
    y_train = train_data[target].values
    X_test = pd.get_dummies(test_data[predictive_cols], drop_first=True)
    y_test = test_data[target].values
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    scaler_panel = StandardScaler()
    X_train_scaled = scaler_panel.fit_transform(X_train)
    X_test_scaled = scaler_panel.transform(X_test)
    
    results = {}  # Pour stocker uniquement le MSE sur test

    # --- Modèle 1 : ARIMAX (modélisation par pays) ---
    arimax_test_preds = []
    arimax_true_test = []
    arimax_forecast_dict = {}
    
    exog_cols = [
        'Year',
        'Adjusted net national income (current US$)',
        'Birth rate, crude (per 1,000 people)',
        'Death rate, crude (per 1,000 people)',
        'Exports of goods and services (% of GDP)',
        'GDP per capita (current US$)',
        'Life expectancy at birth, total (years)',
        'Population growth (annual %)'
    ]
    
    for country in data["Country Name"].unique():
        df_train = train_data[train_data["Country Name"] == country].sort_values("Year")
        df_test = test_data[test_data["Country Name"] == country].sort_values("Year")
        if len(df_train) < 10:
            continue
        endog_train = df_train[target]
        endog_test = df_test[target]
        exog_train = df_train[exog_cols]
        exog_test = df_test[exog_cols]
        try:
            model = ARIMA(endog=endog_train, exog=exog_train, order=(1,1,1))
            fit_model = model.fit()
            pred_test = fit_model.predict(start=len(endog_train),
                                          end=len(endog_train)+len(endog_test)-1,
                                          exog=exog_test)
            arimax_test_preds.extend(pred_test)
            arimax_true_test.extend(endog_test)
            
            # Prévision future : ré-ajustement sur toutes les données disponibles pour le pays
            df_full = data[data["Country Name"] == country].sort_values("Year")
            endog_full = df_full[target]
            exog_full = df_full[exog_cols]
            model_full = ARIMA(endog=endog_full, exog=exog_full, order=(1,1,1))
            fit_full = model_full.fit()
            n_forecast = len(forecast_years)
            last_exog = df_full[exog_cols].iloc[-1].copy()
            forecast_exog = pd.DataFrame({"Year": forecast_years})
            for col in exog_cols:
                if col != "Year":
                    forecast_exog[col] = last_exog[col]
            forecast_exog = forecast_exog[exog_cols]
            forecast_pred = fit_full.get_forecast(steps=n_forecast, exog=forecast_exog).predicted_mean
            arimax_forecast_dict[country] = forecast_pred
        except Exception as e:
            print(f"ARIMAX échoue pour {country}: {e}")
    
    mse_arimax_test = mean_squared_error(arimax_true_test, arimax_test_preds)
    results["ARIMAX"] = {"mse_test": mse_arimax_test}
    arimax_forecast_df = pd.DataFrame(arimax_forecast_dict)
    arimax_forecast_agg = arimax_forecast_df.sum(axis=1).values

    # --- Modèle 2 : RandomForest ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_test_pred = rf.predict(X_test_scaled)
    mse_rf_test = mean_squared_error(y_test, rf_test_pred)
    results["RandomForest"] = {"mse_test": mse_rf_test}
    
    forecast_panel_list = []
    for country in full_data["Country Name"].unique():
        df_country = full_data[full_data["Country Name"] == country].sort_values("Year")
        if df_country.empty:
            continue
        last_row = df_country.iloc[-1].copy()
        for yr in forecast_years:
            new_row = last_row.copy()
            new_row["Year"] = yr
            forecast_panel_list.append(new_row)
    forecast_panel_df = pd.DataFrame(forecast_panel_list)
    X_forecast = pd.get_dummies(forecast_panel_df[predictive_cols], drop_first=True)
    X_forecast = X_forecast.reindex(columns=X_train.columns, fill_value=0)
    X_forecast_scaled = scaler_panel.transform(X_forecast)
    rf_forecast_pred = rf.predict(X_forecast_scaled)
    forecast_panel_df["RF_Pred"] = rf_forecast_pred
    rf_forecast_agg = forecast_panel_df.groupby("Year")["RF_Pred"].sum().values

    # --- Modèle 3 : XGBoost ---
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train_scaled, y_train)
    xgb_test_pred = xgb.predict(X_test_scaled)
    mse_xgb_test = mean_squared_error(y_test, xgb_test_pred)
    results["XGBoost"] = {"mse_test": mse_xgb_test}
    xgb_forecast_pred = xgb.predict(X_forecast_scaled)
    forecast_panel_df["XGB_Pred"] = xgb_forecast_pred
    xgb_forecast_agg = forecast_panel_df.groupby("Year")["XGB_Pred"].sum().values

    # --- Modèle 4 : Ridge ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_test_pred = ridge.predict(X_test_scaled)
    mse_ridge_test = mean_squared_error(y_test, ridge_test_pred)
    results["Ridge"] = {"mse_test": mse_ridge_test}
    ridge_forecast_pred = ridge.predict(X_forecast_scaled)
    forecast_panel_df["Ridge_Pred"] = ridge_forecast_pred
    ridge_forecast_agg = forecast_panel_df.groupby("Year")["Ridge_Pred"].sum().values

    # --- Modèle 5 : LSTM ---
    train_sorted = train_data.sort_values(["Country Name", "Year"])
    scaler_y = StandardScaler()
    X_train_panel = pd.get_dummies(train_sorted[predictive_cols], drop_first=True)
    X_train_panel = X_train_panel.reindex(columns=X_train.columns, fill_value=0)
    X_train_panel_arr = scaler_panel.transform(X_train_panel)
    y_train_panel = train_sorted[target].values
    y_train_panel_scaled = scaler_y.fit_transform(y_train_panel.reshape(-1,1))
    
    window_size = 3
    lstm_X_seq = []
    lstm_y_seq = []
    for country in train_sorted["Country Name"].unique():
        df_country = train_sorted[train_sorted["Country Name"] == country].sort_values("Year")
        X_country = scaler_panel.transform(pd.get_dummies(df_country[predictive_cols], drop_first=True).reindex(columns=X_train.columns, fill_value=0))
        y_country = scaler_y.transform(df_country[target].values.reshape(-1,1))
        for i in range(len(X_country) - window_size):
            lstm_X_seq.append(X_country[i:i+window_size])
            lstm_y_seq.append(y_country[i+window_size])
    lstm_X_seq = np.array(lstm_X_seq)
    lstm_y_seq = np.array(lstm_y_seq)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation="relu", input_shape=(window_size, lstm_X_seq.shape[2])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(lstm_X_seq, lstm_y_seq, epochs=50, verbose=0)
    
    test_sorted = test_data.sort_values(["Country Name", "Year"])
    lstm_X_seq_test = []
    lstm_y_seq_test = []
    for country in test_sorted["Country Name"].unique():
        df_country = test_sorted[test_sorted["Country Name"] == country].sort_values("Year")
        if len(df_country) <= window_size:
            continue
        X_country = scaler_panel.transform(pd.get_dummies(df_country[predictive_cols], drop_first=True).reindex(columns=X_train.columns, fill_value=0))
        y_country = scaler_y.transform(df_country[target].values.reshape(-1,1))
        for i in range(len(X_country) - window_size):
            lstm_X_seq_test.append(X_country[i:i+window_size])
            lstm_y_seq_test.append(y_country[i+window_size])
    lstm_X_seq_test = np.array(lstm_X_seq_test)
    lstm_y_seq_test = np.array(lstm_y_seq_test)
    
    lstm_test_pred_scaled = lstm_model.predict(lstm_X_seq_test)
    lstm_test_pred = scaler_y.inverse_transform(lstm_test_pred_scaled)
    true_y_test = scaler_y.inverse_transform(lstm_y_seq_test)
    mse_lstm_test = mean_squared_error(true_y_test, lstm_test_pred)
    results["LSTM"] = {"mse_test": mse_lstm_test}
    
    # Prévision LSTM itérative par pays
    lstm_forecast_dict = {}
    for country in full_data["Country Name"].unique():
        df_country = full_data[full_data["Country Name"] == country].sort_values("Year")
        X_country = scaler_panel.transform(pd.get_dummies(df_country[predictive_cols], drop_first=True).reindex(columns=X_train.columns, fill_value=0))
        if len(X_country) < window_size:
            continue
        last_window = X_country[-window_size:]
        forecasts = []
        for _ in range(len(forecast_years)):
            input_seq = last_window.reshape(1, window_size, -1)
            pred_scaled = lstm_model.predict(input_seq)
            pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
            forecasts.append(pred)
            new_input = X_country[-1]
            last_window = np.vstack([last_window[1:], new_input])
        lstm_forecast_dict[country] = forecasts
    lstm_forecast_df = pd.DataFrame(lstm_forecast_dict, index=forecast_years)
    lstm_forecast_agg = lstm_forecast_df.sum(axis=1).values

    # --- Affichage du MSE sur test ---
    print("MSE sur test :")
    for model_name, metrics in results.items():
        print(f"{model_name}: MSE Test = {metrics['mse_test']:.2f}")
    
    # --- Graphique de la population mondiale (1960 à 2050) ---
    global_actual = data.groupby("Year")[target].sum().reset_index()
    plt.figure(figsize=(12,6))
    plt.plot(global_actual["Year"], global_actual[target], label="Données réelles", marker="o")
    plt.plot(forecast_years, arimax_forecast_agg, label="ARIMAX", linestyle="--")
    plt.plot(forecast_years, rf_forecast_agg, label="RandomForest", linestyle="--")
    plt.plot(forecast_years, xgb_forecast_agg, label="XGBoost", linestyle="--")
    plt.plot(forecast_years, ridge_forecast_agg, label="Ridge", linestyle="--")
    plt.plot(forecast_years, lstm_forecast_agg, label="LSTM", linestyle="--")
    plt.xlabel("Année")
    plt.ylabel("Population mondiale")
    plt.title("Population mondiale de 1960 à 2050\n(prédictions à partir de 2024)")
    plt.legend()
    plt.grid(True)
    plt.xlim(1960, 2050)
    plt.savefig("population_forecast.png")
    plt.close()
    
    # --- Graphique comparatif des MSE sur test ---
    model_names = list(results.keys())
    mse_test_values = [results[m]["mse_test"] for m in model_names]
    x_axis = np.arange(len(model_names))
    width = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x_axis, mse_test_values, width, label="MSE Test")
    plt.xticks(x_axis, model_names)
    plt.xlabel("Modèles")
    plt.ylabel("MSE")
    plt.title("Comparaison des MSE sur test")
    plt.legend()
    plt.grid(True)
    plt.savefig("mse_comparison.png")
    plt.close()

if __name__ == "__main__":
    compare_models_and_plot()
