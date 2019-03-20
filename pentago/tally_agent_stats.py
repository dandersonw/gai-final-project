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
    parser.add_argument('--trials', default=100)
    args = parser.parse_args()

    agents = [pentago.get_agent_for_str(None,
                                        args.white_agent,
                                        pentago.WHITE_TURN,
                                        **json.loads(args.white_config)),
              pentago.get_agent_for_str(None,
                                        args.black_agent,
                                        pentago.BLACK_TURN,
                                        **json.loads(args.black_config))]

    black_wins = 0
    white_wins = 0
    draws = 0
    for i in tqdm(list(range(args.trials))):
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