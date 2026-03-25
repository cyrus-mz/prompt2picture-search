from src.image_db import create_image_db

if __name__ == "__main__":
    print("Scanning image folder and creating metadata DB...")
    create_image_db()
    print("Done: image_db.json created/updated.")
