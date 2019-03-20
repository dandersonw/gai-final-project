import numpy as np

import abc
import typing

from . import game, search_ai


class Agent(abc.ABC):
    @abc.abstractmethod
    def make_move(self, game: game.Game):
        pass

    @abc.abstractmethod
    def load_params(self, config: typing.Dict[str, str]):
        pass


def get_agent_for_str(human, string, color_turn, **kwargs) -> Agent:
    if string == 'human':
        return human
    elif string == 'random':
        return RandomAgent(color_turn)
    elif string == 'minimax':
        return MinimaxSearchAgent(color_turn, **kwargs)


class AIAgent(Agent):
    def __init__(self, player_no):
        self.player_no = player_no

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

    def load_params(self, config):
        return dict()


class MinimaxSearchAgent(AIAgent):
    def __init__(self, player_no, *, depth=1, evaluation_function=search_ai.runs_evaluation):
        super(MinimaxSearchAgent, self).__init__(player_no)
        self.depth = depth
        self.evaluation_function = evaluation_function

    def load_params(self, config):
        result = {k: v for k, v in config.items() if k in {'depth'}}
        ef_key = config.get('evaluation_function')
        if ef_key is not None:
            result['evaluation_function'] = search_ai.evaluation_function_for_str(ef_key)
        return result

    def _strategy(self, board):
        return search_ai.minimax_search(board,
                                        self.evaluation_function,
                                        self.depth)
