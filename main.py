import argparse
import sys

from processes import train, test


def main():
    parser = argparse.ArgumentParser(description='Differentiable Programming')
    subparsers = parser.add_subparsers(title='Actions', dest='action')

    train.set_parser(subparsers)
    test.set_parser(subparsers)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)

    if not args:
        print('Invalid command, for help use --help')
        parser.parse_args(['-h'])
        sys.exit(2)


if __name__ == '__main__':
    main()
