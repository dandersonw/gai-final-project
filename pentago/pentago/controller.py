import abc
import typing

from . import agent, game, render


class View(abc.ABC):
    @abc.abstractmethod
    def render(model: game.Game):
        pass

    @abc.abstractmethod
    def game_ended(winner: int):
        pass


class DumbTextView(View):
    def render(model: game.Game):
        print(render.render_board(model))
        print('-' * 20)

    def game_ended(winner):
        if winner == 0:
            print('DRAW!')
        elif winner == 1:
            print('WHITE WINS!')
        elif winner == -1:
            print('BLACK WINS!')


class Controller():
    def __init__(self, agents: typing.List[agent.Agent], view: View):
        self.agents = agents
        self.view = view

    def play_game(self):
        model = game.Game()
        winner = None
        while winner is None:
            for agentt in self.agents:
                self.view.render(model)
                agentt.make_move(model)
                winner = model.check_game_over()
                if winner is not None:
                    break
        self.view.game_ended(winner)
