import argparse
import json
import pentago

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--black-agent', default='random')
    parser.add_argument('--black-config', default='{}')
    parser.add_argument('--white-agent', default='human')
    parser.add_argument('--white-config', default='{}')
    parser.add_argument('--trials', default=100, type=int)
    args = parser.parse_args()

    agents = [pentago.get_agent_for_key(None,
                                        args.white_agent,
                                        json.loads(args.white_config)),
              pentago.get_agent_for_key(None,
                                        args.black_agent,
                                        json.loads(args.black_config))]

    black_wins = 0
    white_wins = 0
    draws = 0
    trials = tqdm(list(range(args.trials)))
    for i in trials:
        trials.set_description('B/W/D: {}/{}/{}'.format(black_wins,
                                                        white_wins,
                                                        draws))
        controller = pentago.Controller(agents, None)
        winner = controller.play_game()
        if winner == 1:
            white_wins += 1
        elif winner == -1:
            black_wins += 1
        else:
            draws += 1

    print('White wins: {}, Black wins: {}, Draws: {}'
          .format(white_wins, black_wins, draws))


if __name__ == '__main__':
    main()
