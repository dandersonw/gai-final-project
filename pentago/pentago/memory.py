import typing

from collections import deque

from . import game, agent, view


class Experience():
    def __init__(self, model: game.Game):
        self.turn = model.turn
        self.board = model.board * model.turn

    def add_explanation(self, explanation: agent.Explanation):
        self.explanation = explanation

    def game_ended(self, winner: int):
        self.value = self.turn * winner


class MemoryView(view.View):
    def __init__(self):
        self.memory = deque()
        self.temp_memory = deque()

    def render(self, model: game.Game):
        self.temp_memory.append(Experience(model))

    def move_made(self, move: game.Move, explanation: agent.Explanation):
        self.temp_memory[-1].add_explanation(explanation)

    def game_ended(self, winner: int):
        for exp in self.temp_memory:
            exp.game_ended(winner)
            self.memory.append(exp)
        self.temp_memory.clear()

    def get_experiences(self) -> typing.List[Experience]:
        return list(self.memory)
