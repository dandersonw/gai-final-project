import numpy as np

import abc
import typing

from collections import deque

from . import game


Explanation = typing.Any


class Agent(abc.ABC):
    @property
    @abc.abstractstaticmethod
    def key():
        return None

    @abc.abstractmethod
    def make_move(self, game: game.Game) -> Explanation:
        """Make a move and return an explanation."""
        pass

    @abc.abstractclassmethod
    def load_params(cls, config: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        """From a serialized key:value representation, load params of the type that the
        constructor for this class wants.

        """
        pass

    @abc.abstractmethod
    def to_params(self) -> typing.Dict[str, typing.Any]:
        """Produce a serializable key:value representation of this agent."""
        pass


def get_agent_for_key(human, key, config=dict()) -> Agent:
    """Search through extant subclasses of Agent for the one that matches key and
    then instantiate that agent with config.

    """
    if key == 'human':
        return human

    q = deque()
    q.append(Agent)
    while q:
        a: Agent = q.pop()
        if a.key == key:
            params = a.load_params(config)
            return a(**params)
        q.extend(a.__subclasses__())

    raise NameError('Can\'t find agent of key {}'.format(key))


class AIAgent(Agent):
    def make_move(self, game: game.Game) -> typing.Tuple[game.Move, Explanation]:
        return self._strategy(game.board * game.turn)

    @abc.abstractmethod
    def _strategy(self, board: game.Board) -> typing.Tuple[game.Move, Explanation]:
        """Decide on a placement and a rotation.

        All values in board are from the perspective of this player, i.e. 1
        is a piece of this player's color and -1 is a piece of the other
        player's color

        """
        pass


class RandomAgent(AIAgent):
    key = 'random'

    def _strategy(self, board: game.Board) -> typing.Tuple[game.Move, Explanation]:
        can_place_mask = game.can_place_mask(board)
        num_possible = np.sum(can_place_mask)
        possible_moves = np.argsort(np.ravel(can_place_mask))[-num_possible:]
        np.random.shuffle(possible_moves)
        move = possible_moves[0]
        indices = np.unravel_index(move, game.BOARD_DIMS)
        placement = game.generate_placement_from_indices(*indices)

        rotation_quadrant = np.random.randint(4)
        rotation_dir = np.random.randint(2) - 1
        rotation = np.zeros([4], dtype=np.int8)
        rotation[rotation_quadrant] = rotation_dir
        rotation = np.reshape(rotation, [2, 2])

        return (placement, rotation), None

    @classmethod
    def load_params(cls, config):
        return dict()

    def to_params(self):
        return dict()
