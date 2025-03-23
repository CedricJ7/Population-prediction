import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# Chargement et préparation des données
# -------------------------------
data = pd.read_csv("data/data_countries_imputed_iterative.csv")
predictive_col = [
    'Year',
    'Adjusted net national income (current US$)',
    'Birth rate, crude (per 1,000 people)',
    'Death rate, crude (per 1,000 people)',
    'Exports of goods and services (% of GDP)',
    'GDP per capita (current US$)',
    'Life expectancy at birth, total (years)',
    'Population growth (annual %)',
    "Country Name",
    "Continent"
]

# Séparation en jeu d'entraînement (année < 2000) et de test (année >= 2000)
data_train = data[data["Year"] < 2000].copy()
data_test  = data[data["Year"] >= 2000].copy()

# Variable cible
y_col = "Population, total"
y_train = data_train[y_col]
y_test  = data_test[y_col]

# Création des matrices explicatives
X_train = data_train[predictive_col]
X_test  = data_test[predictive_col]

# Sélection des colonnes numériques
numeric_cols = [
    'Year',
    'Adjusted net national income (current US$)',
    'Birth rate, crude (per 1,000 people)',
    'Death rate, crude (per 1,000 people)',
    'Exports of goods and services (% of GDP)',
    'GDP per capita (current US$)',
    'Life expectancy at birth, total (years)',
    'Population growth (annual %)'
]
X_train = X_train[numeric_cols]
X_test  = X_test[numeric_cols]

# Standardisation des variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -------------------------------
# Construction et entraînement du réseau de neurones
# -------------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],),
          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Couche de sortie pour la régression
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train,
                    epochs=1000,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1)

# -------------------------------
# Affichage de l'évolution du RMSE pendant l'entraînement
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(history.history['root_mean_squared_error'], label='RMSE entraînement')
plt.plot(history.history['val_root_mean_squared_error'], label='RMSE validation')
plt.xlabel('Époch')
plt.ylabel('RMSE')
plt.title("Évolution de la fonction de coût (RMSE)")
plt.legend()
plt.show()

# -------------------------------
# Prédiction pour les années futures
# -------------------------------
# Définir les années futures souhaitées, par exemple de 2023 à 2032
future_years = np.arange(2023, 2033)

# On suppose que les indicateurs futurs restent constants et égaux à la moyenne du jeu de test
avg_values = pd.DataFrame(X_test, columns=numeric_cols).mean()

future_df = pd.DataFrame({
    'Year': future_years,
    'Adjusted net national income (current US$)': avg_values['Adjusted net national income (current US$)'],
    'Birth rate, crude (per 1,000 people)': avg_values['Birth rate, crude (per 1,000 people)'],
    'Death rate, crude (per 1,000 people)': avg_values['Death rate, crude (per 1,000 people)'],
    'Exports of goods and services (% of GDP)': avg_values['Exports of goods and services (% of GDP)'],
    'GDP per capita (current US$)': avg_values['GDP per capita (current US$)'],
    'Life expectancy at birth, total (years)': avg_values['Life expectancy at birth, total (years)'],
    'Population growth (annual %)': avg_values['Population growth (annual %)']
})

# Standardisation des données futures avec le même scaler
future_scaled = scaler.transform(future_df)
future_pred = model.predict(future_scaled)

# -------------------------------
# Graphique combiné : Données historiques et prédictions futures
# -------------------------------
# Extraire les données historiques (années et population réelle)
historical_years = data['Year']
historical_population = data[y_col]

plt.figure(figsize=(10, 6))
plt.plot(historical_years, historical_population, label='Données historiques', marker='o', linestyle='-')
plt.plot(future_years, future_pred, label='Prédictions futures', marker='o', linestyle='--', color='red')
plt.xlabel('Année')
plt.ylabel('Population')
plt.title("Population historique et prédictions futures")
plt.legend()
plt.show()
