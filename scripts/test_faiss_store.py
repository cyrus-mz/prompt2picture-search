from src.models.clip_model import encode_text
from src.storage.faiss_store import search_index, get_record_by_faiss_index


def main():
    query = 'a dog running in the grass'
    query_vector = encode_text(query)

    distances, indices = search_index(query_vector, top_k=3)

    print(f'Query: {query}')
    print()

    for rank, (distance, faiss_index) in enumerate(zip(distances, indices), start=1):
        record = get_record_by_faiss_index(int(faiss_index))
        print(f'Rank {rank}')
        print(f'  Distance: {float(distance):.6f}')
        print(f'  Image ID: {record["image_id"]}')
        print(f'  Filename: {record["filename"]}')
        print(f'  Path: {record["path"]}')
        print()


if __name__ == '__main__':
    main()