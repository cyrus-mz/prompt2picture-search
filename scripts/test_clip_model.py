from pathlib import Path

from src.models.clip_model import encode_text, encode_image
from src.config import IMAGE_FOLDER


def main():
    text_embedding = encode_text("This is a test for encoding text")
    print(f"Text embedding shape: {text_embedding.shape}")
    print(f"Text embedding dtype: {text_embedding.dtype}")

    image_files = list(Path(IMAGE_FOLDER).glob("*.jpg"))
    if not image_files:
        print(f"No images found in {IMAGE_FOLDER}")
        return
    

    image_embedding = encode_image(image_files[0])
    print(f'Image embedding shape: {image_embedding.shape}')
    print(f'Image embedding dtype: {image_embedding.dtype}')


if __name__ == "__main__":
    main()

