"""
WSGI Entry point for the Dash application.
This file is used by gunicorn to properly serve the application on Render.
"""

import os
from app import server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    server.run(host="0.0.0.0", port=port) 