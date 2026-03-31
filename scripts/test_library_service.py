import argparse
from pprint import pprint

from src.services.library_service import scan_indexed_library, rebuild_index_from_db


def main():
    parser = argparse.ArgumentParser(description='Test library scan and rebuild service')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('scan', help='Scan indexed library paths')

    rebuild_parser = subparsers.add_parser('rebuild', help='Rebuild index from database')
    rebuild_parser.add_argument('--batch-size', type=int, default=16, help='Batch size for re-embedding')

    args = parser.parse_args()

    if args.command == 'scan':
        report = scan_indexed_library()

        print('\n=== LIBRARY SCAN REPORT ===\n')

        print(f'Available: {len(report["available"])}')
        print(f'Missing:   {len(report["missing"])}')
        print(f'Invalid:   {len(report["invalid"])}')

        if report['missing']:
            print('\nMissing examples:')
            for item in report['missing'][:5]:
                pprint(item)

        if report['invalid']:
            print('\nInvalid examples:')
            for item in report['invalid'][:5]:
                pprint(item)

    elif args.command == 'rebuild':
        report = rebuild_index_from_db(batch_size=args.batch_size)

        print('\n=== REBUILD REPORT ===\n')

        print(f'Reindexed: {len(report["reindexed"])}')
        print(f'Missing:   {len(report["missing"])}')
        print(f'Failed:    {len(report["failed"])}')

        if report['missing']:
            print('\nMissing examples:')
            for item in report['missing'][:5]:
                pprint(item)

        if report['failed']:
            print('\nFailed examples:')
            for item in report['failed'][:5]:
                pprint(item)


if __name__ == '__main__':
    main()