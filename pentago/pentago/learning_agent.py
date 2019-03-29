import abc

import typing

from . import agent, memory


class SelfPlayAgent(agent.AIAgent):
    @abc.abstractmethod
    def fit(self, exps: typing.List[memory.Experience]):
        pass


class NeuralAgent(SelfPlayAgent):
    key = 'neural'

    def fit(self, exps: typing.List[memory.Experience]):
        pass

    def _strategy(self, board):
        pass

    @classmethod
    def load_params(cls, config):
        return dict()

    def to_params(self):
        return dict()
