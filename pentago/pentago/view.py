import abc

from . import game, render, agent


class View(abc.ABC):
    @abc.abstractmethod
    def render(self, model: game.Game):
        pass

    @abc.abstractmethod
    def move_made(self, move: game.Move, explanation: agent.Explanation):
        pass

    @abc.abstractmethod
    def game_ended(self, winner: int):
        pass


class DumbTextView(View):
    def render(self, model: game.Game):
        print(render.render_board(model))
        print('-' * 20)

    def move_made(self, move: game.Move, explanation: agent.Explanation):
        print(move)

    def game_ended(self, winner):
        if winner == 0:
            print('DRAW!')
        elif winner == 1:
            print('WHITE WINS!')
        elif winner == -1:
            print('BLACK WINS!')
