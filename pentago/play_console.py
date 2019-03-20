import argparse
import pentago


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    with pentago.ConsoleClient(pentago.WHITE_TURN) as client:
        ai_opponent = pentago.RandomAgent(pentago.BLACK_TURN)
        controller = pentago.Controller([client, ai_opponent], client)
        controller.play_game()


if __name__ == '__main__':
    main()
