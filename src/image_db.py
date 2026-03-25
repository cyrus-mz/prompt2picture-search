import os
import json
import uuid
from src.config import IMAGE_FOLDER, DB_FILE

def load_image_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_image_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

def create_image_db():
    db = load_image_db()
    existing_filenames = {entry["filename"] for entry in db.values()}

    new_files = [
        fname for fname in os.listdir(IMAGE_FOLDER)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        and fname not in existing_filenames
    ]

    for fname in new_files: 
        image_id = str(uuid.uuid4())[:8]
        db[image_id] = {
            "filename": fname,
            "tags": [],
            "caption": "",
            "embedding": None
        }

    save_image_db(db)
    print(f"Image DB created/updated with {len(new_files)} new entries.")

    return db
