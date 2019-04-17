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
                        'Expects the path to a pickle file. Supersedes --second-config')
    parser.add_argument('--first-agent', default='human')
    parser.add_argument('--first-config', default='{}')
    parser.add_argument('--first-pickle',
                        help='Config with which to initialize the first agent. '
                        'Expects the path to a pickle file. Supersedes --first-config')
    parser.add_argument('--trials', default=100, type=int)
    args = parser.parse_args()

    if args.first_pickle is None:
        first_config = json.loads(args.first_config)
    else:
        first_config = pickle.load(open(args.first_pickle, mode='rb'))
    if args.second_pickle is None:
        second_config = json.loads(args.second_config)
    else:
        second_config = pickle.load(open(args.second_pickle, mode='rb'))

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
