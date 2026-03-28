from src.services.search_service import search_by_text


def main():
    query = "a dog running in the grass"
    results = search_by_text(query, top_k=5)

    print(f'Query: "{query}"')
    print()

    for result in results:
        print(f'Rank: {result["rank"]}')
        print(f'  Image ID: {result["image_id"]}')
        print(f'  Filename: {result["filename"]}')
        print(f'  Path: {result["path"]}')
        print(f'  Raw distance: {result["raw_distance"]:.6f}')
        print(f'  Similarity score: {result["similarity_score"]:.6f}')
        print()


if __name__ == "__main__":
    main()