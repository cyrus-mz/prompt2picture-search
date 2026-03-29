from pathlib import Path

from src.config import IMAGE_FOLDER
from src.models.clip_model import encode_images_batch, encode_texts_batch


def main():
    text_queries = [
        'a dog running in the grass',
        'a mountain at sunset',
    ]
    text_embeddings = encode_texts_batch(text_queries)
    print(f'Text batch shape: {text_embeddings.shape}')
    print(f'Text batch dtype: {text_embeddings.dtype}')

    image_files = list(Path(IMAGE_FOLDER).glob('*.jpg'))[:2]
    if len(image_files) < 2:
        print('Not enough images found in data/images')
        return

    image_embeddings = encode_images_batch(image_files)
    print(f'Image batch shape: {image_embeddings.shape}')
    print(f'Image batch dtype: {image_embeddings.dtype}')


if __name__ == '__main__':
    main()