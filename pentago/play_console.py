import argparse
import pentago


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--black-agent', default='random')
    parser.add_argument('--white-agent', default='human')
    args = parser.parse_args()

    with pentago.ConsoleClient() as client:
        agents = [get_agent_for_str(client, args.white_agent, pentago.WHITE_TURN),
                  get_agent_for_str(client, args.black_agent, pentago.BLACK_TURN)]
        controller = pentago.Controller(agents, client)
        controller.play_game()


def get_agent_for_str(client, string, color_turn):
    if string == 'human':
        return client
    elif string == 'random':
        return pentago.RandomAgent(color_turn)


if __name__ == '__main__':
    main()
