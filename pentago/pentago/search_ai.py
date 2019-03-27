import numpy as np

import numba

from . import game, agent


def evaluation_function_for_str(string):
    if string == 'runs':
        return runs_evaluation


def minimax_search(board, evaluation_function, depth=3):
    val, move = _minimax_search(board, depth, 1, evaluation_function)
    return move


@numba.jit(nopython=True)
def _minimax_search(board, depth, sign, evaluation_function):
    if game.check_game_over(board) is not None or depth == 0:
        return evaluation_function(board), None

    best_move = None
    best_val = 0
    for i in np.random.permutation(game.BOARD_H):
        for j in np.random.permutation(game.BOARD_W):
            if board[i, j] != 0:
                continue
            placement = game.generate_placement_from_indices(i, j)
            for q_i in np.random.permutation(2):
                for q_j in np.random.permutation(2):
                    for d in [-1, 1]:
                        rotation = game.generate_rotation(q_i, q_j, d)
                        new_board = game.apply_move(board, sign, placement, rotation)
                        val, _ = _minimax_search(new_board,
                                                 depth - 1,
                                                 -1 * sign,
                                                 evaluation_function)
                        if best_move is None or val * sign > best_val * sign:
                            best_val = val
                            best_move = placement, rotation
    return best_val, best_move


@numba.jit(nopython=True, cache=True)
def runs_evaluation(board):
    return count_runs(board, 1) - count_runs(board, -1)


@numba.jit(nopython=True, cache=True)
def count_runs(board, sign):
    runs = np.zeros((6,), dtype=np.float64)
    # check rows
    for i in range(game.BOARD_H):
        _check_line(board, runs, i, 0, 0, 1, sign)

    # check columns
    for j in range(game.BOARD_W):
        _check_line(board, runs, 0, j, 1, 0, sign)

    # check diagonals
    _check_line(board, runs, 0, 0, 1, 1, sign)
    _check_line(board, runs, 1, 0, 1, 1, sign)
    _check_line(board, runs, 0, 1, 1, 1, sign)
    _check_line(board, runs, 5, 0, -1, 1, sign)
    _check_line(board, runs, 4, 0, -1, 1, sign)
    _check_line(board, runs, 5, 1, -1, 1, sign)

    point_values = np.array((0, 2 ** 1, 2 ** 4, 2 ** 6, 2 ** 8, 2 ** 15),
                            dtype=np.float64)
    return np.dot(runs, point_values)


@numba.jit(nopython=True, cache=True)
def _check_line(board, runs, i, j, i_stride, j_stride, sign):
    run = 0
    while i < game.BOARD_H and j < game.BOARD_W:
        if board[i, j] != sign:
            runs[run] += 1
            run = 0
        else:
            run += 1
        i += i_stride
        j += j_stride
    runs[run] += 1


class MinimaxSearchAgent(agent.AIAgent):
    key = 'minimax'

    def __init__(self, *, depth=1, evaluation_function=runs_evaluation):
        super(MinimaxSearchAgent, self).__init__()
        self.depth = depth
        self.evaluation_function = evaluation_function

    def load_params(self, config):
        result = {k: v for k, v in config.items() if k in {'depth'}}
        ef_key = config.get('evaluation_function')
        if ef_key is not None:
            result['evaluation_function'] = evaluation_function_for_str(ef_key)
        return result

    def _strategy(self, board):
        return minimax_search(board,
                              self.evaluation_function,
                              self.depth)
