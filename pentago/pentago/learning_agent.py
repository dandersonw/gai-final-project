import abc

import typing

from . import agent, memory


class SelfPlayAgent(agent.AIAgent):
    @abc.abstractmethod
    def fit(self, exps: typing.List[memory.Experience]):
        pass
