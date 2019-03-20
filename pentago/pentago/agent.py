import numpy as np

import abc

from . import game


class Agent(abc.ABC):
    def __init__(self, player_no):
        self.player_no = player_no

    @abc.abstractmethod
    def make_move(self, game: game.Game):
        pass


class AIAgent(Agent):
    def make_move(self, game: game.Game):
        move = self._strategy(game.board * self.player_no)
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
    def _strategy(self, board: game.Board) -> game.Move:
        can_place_mask = game.can_place_mask(board)
        possible_moves = np.argsort(np.ravel(can_place_mask))[-np.sum(can_place_mask):]
        np.random.shuffle(possible_moves)
        move = possible_moves[0]
        placement = game.generate_placement_from_indices(*np.unravel_index(move, game.BOARD_DIMS))

        rotation_quadrant = np.random.randint(4)
        rotation_dir = np.random.randint(2) - 1
        rotation = np.zeros([4], dtype=np.int8)
        rotation[rotation_quadrant] = rotation_dir
        rotation = np.reshape(rotation, [2, 2])

        return placement, rotation
