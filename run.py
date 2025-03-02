"""
Script principal pour lancer l'application de prédiction de population mondiale.
Ce script importe l'application du fichier app.py et la lance.
"""

import webbrowser
import threading
import time
import os

# Importation correcte de l'objet app
from app import app, server

# Fonction pour ouvrir automatiquement le navigateur
def open_browser():
    """Ouvre le navigateur par défaut avec l'URL de l'application après un court délai."""
    # Attendre 1 seconde pour que le serveur ait le temps de démarrer
    time.sleep(1)
    # Ouvrir l'URL dans le navigateur par défaut
    webbrowser.open_new("http://127.0.0.1:8050/")

# Lancer l'application
if __name__ == '__main__':
    # Créer le dossier assets s'il n'existe pas
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    # Créer le dossier templates s'il n'existe pas
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Démarrer un thread qui ouvrira le navigateur
    threading.Thread(target=open_browser).start()
    
    # Démarrer l'application Dash
    app.run_server(debug=True)