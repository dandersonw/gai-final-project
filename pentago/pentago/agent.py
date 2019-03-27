import numpy as np

import abc
import typing

from collections import deque

from . import game


class Agent(abc.ABC):
    @property
    @abc.abstractstaticmethod
    def key():
        return None

    @abc.abstractmethod
    def make_move(self, game: game.Game):
        pass

    @abc.abstractmethod
    def load_params(self, config: typing.Dict[str, str]):
        pass


def get_agent_for_key(human, key, **kwargs) -> Agent:
    if key == 'human':
        return human

    q = deque()
    q.append(Agent)
    while q:
        a = q.pop()
        if a.key == key:
            return a(**kwargs)
        q.extend(a.__subclasses__())

    raise NameError('Can\'t find agent of key {}'.format(key))


class AIAgent(Agent):
    def make_move(self, game: game.Game):
        move = self._strategy(game.board * game.turn)
        game.make_move(move)

    @abc.abstractmethod
    def _strategy(board: game.Board) -> game.Move:
        """Decide on a placement and a rotation.

        All values in board are from the perspective of this player, i.e. 1
        is a piece of this player's color and -1 is a piece of the other
        player's color

        """
        pass


class RandomAgent(AIAgent):
    key = 'random'

    def _strategy(self, board: game.Board) -> game.Move:
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

        return placement, rotation

    def load_params(self, config):
        return dict()
