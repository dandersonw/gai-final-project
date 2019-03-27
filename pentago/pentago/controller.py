import abc
import typing

from collections import deque

from . import agent, game, render


class View(abc.ABC):
    @abc.abstractmethod
    def render(self, model: game.Game):
        pass

    @abc.abstractmethod
    def game_ended(self, winner: int):
        pass


class DumbTextView(View):
    def render(self, model: game.Game):
        print(render.render_board(model))
        print('-' * 20)

    def game_ended(self, winner):
        if winner == 0:
            print('DRAW!')
        elif winner == 1:
            print('WHITE WINS!')
        elif winner == -1:
            print('BLACK WINS!')


class MemoryView(View):
    def __init__(self):
        self.memory = deque()
        self.temp_memory = deque()

    def render(self, model: game.Game):
        self.temp_memory.append({'turn': model.turn,
                                 'board': model.board,
                                 'value': None,  # populated by game_ended
                                 'action_values': None})  # TODO

    def game_ended(self, winner):
        for exp in self.temp_memory:
            exp['value'] = exp['turn'] * winner
            self.memory.append(exp)
        self.temp_memory = deque()


class Controller():
    def __init__(self, agents: typing.List[agent.Agent],
                 view: typing.Optional[View]):
        self.agents = agents
        self.view = view

    def play_game(self):
        model = game.Game()
        winner = None
        while winner is None:
            for agentt in self.agents:
                if self.view is not None:
                    self.view.render(model)
                agentt.make_move(model)
                winner = model.check_game_over()
                if winner is not None:
                    break
        if self.view is not None:
            self.view.game_ended(winner)
        return winner
