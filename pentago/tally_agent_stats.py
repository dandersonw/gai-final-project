import argparse
import json
import pentago
import pentago.evaluate

import dill as pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--second-agent', default='random')
    parser.add_argument('--second-config',
                        default='{}',
                        help='Config with which to initialize the second agent.')
    parser.add_argument('--second-pickle',
                        help='Config with which to initialize the second agent. '
                        'Expects the path to a pickle file. Superseded by --second-config')
    parser.add_argument('--first-agent', default='human')
    parser.add_argument('--first-config', default='{}')
    parser.add_argument('--first-pickle',
                        help='Config with which to initialize the first agent. '
                        'Expects the path to a pickle file. Superseded by --first-config')
    parser.add_argument('--trials', default=100, type=int)
    args = parser.parse_args()

    first_config = json.loads(args.first_config)
    if args.first_pickle is not None:
        first_config = {**pickle.load(open(args.first_pickle, mode='rb')),
                        **first_config}
    second_config = json.loads(args.second_config)
    if args.second_pickle is not None:
        second_config = {**pickle.load(open(args.second_pickle, mode='rb')),
                         **second_config}

    agents = [pentago.get_agent_for_key(None,
                                        args.first_agent,
                                        first_config),
              pentago.get_agent_for_key(None,
                                        args.second_agent,
                                        second_config)]

    result = pentago.evaluate.tally_wins(agents,
                                         names=['first', 'second'],
                                         verbose=True,
                                         trials=args.trials)

    print('First wins: {}, Second wins: {}, Draws: {}'
          .format(result[1], result[-1], result[0]))


if __name__ == '__main__':
    main()
