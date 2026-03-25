from src.embedder import embed_all_images
from src.indexer import build_faiss_index

if __name__ == "__main__":
    print("Starting image embedding and indexing...")

    embeddings, ids = embed_all_images(batch_size=32)  # You can change batch size here

    if embeddings:
        build_faiss_index(embeddings)
        print("Done: Embedding + Indexing completed.")
    else:
        print("No new images to embed.")
