import os

# Base folders
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_FOLDER = os.path.join(DATA_DIR, "images")


# DB files
DB_FILE = os.path.join(DATA_DIR, "image_db.json")
INDEX_FILE = os.path.join(DATA_DIR, "image_index.faiss")