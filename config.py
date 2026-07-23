import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    INTENTS_FILE = os.environ.get("INTENTS_FILE", "intents.json")
    STUDENTS_FILE = os.environ.get("STUDENTS_DATA_PATH", "students.json")
    SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.6"))
    SENTENCE_MODEL = os.environ.get("SENTENCE_MODEL", "distiluse-base-multilingual-cased-v2")
    FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.environ.get("FLASK_PORT", 5000))
    FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
