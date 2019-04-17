import argparse
import json
import pentago

import dill as pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--black-agent', default='random')
    parser.add_argument('--black-config',
                        default='{}',
                        help='Config with which to initialize the black agent.')
    parser.add_argument('--black-pickle',
                        help='Config with which to initialize the black agent. '
                        'Expects the path to a pickle file. Superseded by --black-config')
    parser.add_argument('--white-agent', default='human')
    parser.add_argument('--white-config', default='{}')
    parser.add_argument('--white-pickle',
                        help='Config with which to initialize the white agent. '
                        'Expects the path to a pickle file. Superseded by --white-config')
    args = parser.parse_args()

    white_config = json.loads(args.white_config)
    if args.white_pickle is not None:
        white_config = {**pickle.load(open(args.white_pickle, mode='rb')),
                        **white_config}
    black_config = json.loads(args.black_config)
    if args.black_pickle is not None:
        black_config = {**pickle.load(open(args.black_pickle, mode='rb')),
                        **black_config}

    with pentago.ConsoleClient() as client:
        agents = [pentago.get_agent_for_key(client,
                                            args.white_agent,
                                            white_config),
                  pentago.get_agent_for_key(client,
                                            args.black_agent,
                                            black_config)]
        controller = pentago.Controller(agents, client)
        controller.play_game()


if __name__ == '__main__':
    main()
