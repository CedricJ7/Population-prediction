"""
Application principale Dash pour la visualisation de données démographiques mondiales.
Ce fichier importe les composants des autres modules et initialise l'application.
"""

import dash
from dash import html, dcc
import os

# Import des composants personnalisés
from layout import create_layout
from callbacks import register_callbacks

# Initialisation de l'application
app = dash.Dash(
    __name__,
    title='Démographie Mondiale',
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True,
    update_title=None
)

# Configuration du serveur
server = app.server

# Définition du layout
app.layout = create_layout(app)

# Enregistrement des callbacks
register_callbacks(app)

# Lancement de l'application
if __name__ == "__main__":
    app.run_server(debug=True) 