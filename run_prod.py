import logging
from waitress import serve
from app import app
from config import Config

if __name__ == "__main__":
    logger = logging.getLogger("waitress")
    logger.setLevel(logging.INFO)
    print(f"Starting Waitress production server on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    serve(app, host=Config.FLASK_HOST, port=Config.FLASK_PORT, threads=4)
