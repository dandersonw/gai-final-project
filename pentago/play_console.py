import argparse
import json
import pentago


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--black-agent', default='random')
    parser.add_argument('--black-config', default='{}')
    parser.add_argument('--white-agent', default='human')
    parser.add_argument('--white-config', default='{}')
    args = parser.parse_args()

    with pentago.ConsoleClient() as client:
        agents = [pentago.get_agent_for_key(client,
                                            args.white_agent,
                                            **json.loads(args.white_config)),
                  pentago.get_agent_for_key(client,
                                            args.black_agent,
                                            **json.loads(args.black_config))]
        controller = pentago.Controller(agents, client)
        controller.play_game()


if __name__ == '__main__':
    main()
