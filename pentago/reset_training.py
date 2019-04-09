import argparse
from pentago import zobrist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-reset-keys', action='store_true', default=False)
    parser.add_argument('--version')
    args = parser.parse_args()

    if args.reset_keys:
        zobrist.load_keys()

    # TODO Add functionality to revert snapshots

    # print(str(args.version) + str(args.reset_keys))


if __name__ == '__main__':
    main()
