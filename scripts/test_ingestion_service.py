import argparse
from pprint import pprint

from src.services.ingestion_service import ingest_images


def main():
    parser = argparse.ArgumentParser(description='Test image ingestion service')
    parser.add_argument('image_paths', nargs='+', help='Paths of images to ingest')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for embedding')
    args = parser.parse_args()

    report = ingest_images(args.image_paths, batch_size=args.batch_size)

    print('\n=== INGESTION REPORT ===\n')

    print('Ingested:')
    for item in report['ingested']:
        pprint(item)

    print('\nSkipped:')
    for item in report['skipped']:
        pprint(item)

    print('\nFailed:')
    for item in report['failed']:
        pprint(item)


if __name__ == '__main__':
    main()