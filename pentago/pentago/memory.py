import random
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
    def __init__(self, memory_length=30000):
        self.memory = deque(maxlen=memory_length)
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

    def get_experiences(self, sample_k=None) -> typing.List[Experience]:
        if sample_k is None:
            return list(self.memory)
        else:
            sample_k = min(len(self.memory), sample_k)
            return random.sample(self.memory, sample_k)

    def _add_experiences(self, experiences):
        """For use in multiprocessing"""
        self.memory.extend(experiences)
