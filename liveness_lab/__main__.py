import argparse
import csv
import json

from .data import read_table
from .evaluation import evaluate, mine, parity


def main(argv=None):
    parser = argparse.ArgumentParser(description='Offline liveness score evaluation')
    commands = parser.add_subparsers(dest='command', required=True)
    search = commands.add_parser('evaluate', help='select an ensemble and evaluate its frozen threshold')
    search.add_argument('--target-fpr', required=True)
    search.add_argument('--max-size', required=True, type=int)
    mining = commands.add_parser('mine', help='rank difficult training samples within each class')
    mining.add_argument('--model', required=True)
    mining.add_argument('--per-class', type=int, default=1)
    check = commands.add_parser('parity', help='compare paired backend scores on the test split')
    check.add_argument('--reference', required=True)
    check.add_argument('--converted', required=True)
    check.add_argument('--threshold', type=float, required=True)
    for command in (search, mining, check):
        command.add_argument('csv_path', metavar='SCORES.csv')
    args = parser.parse_args(argv)
    try:
        with open(args.csv_path, encoding='utf-8-sig', newline='') as stream:
            table = read_table(stream)
        if args.command == 'evaluate':
            report = evaluate(table, args.target_fpr, args.max_size)
        elif args.command == 'mine':
            report = mine(table, args.model, args.per_class)
        else:
            report = parity(table, args.reference, args.converted, args.threshold)
    except (OSError, ValueError, ZeroDivisionError, csv.Error) as error:
        parser.exit(2, f'input error: {error}\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
