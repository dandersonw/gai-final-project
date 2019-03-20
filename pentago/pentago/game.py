import numpy as np

import numba
import typing


# Note: I reserve the right to have these values hardcoded in other locations
# constants are provided for convenience only
BOARD_W = 6
BOARD_H = 6
QUADRANT_W = BOARD_W // 2
QUADRANT_H = BOARD_H // 2
BOARD_DIMS = [BOARD_H, BOARD_W]

WHITE_TURN = 1
BLACK_TURN = -1

Board = np.ndarray
# dims = [BOARD_H, BOARD_W], dtype = np.uint8
# values of 0 represent a blank space
# nonzero values are either WHITE_TURN or BLACK_TURN,
# representing the respective color of marble
# index 0, 0 is the bottom left corner
# index BOARD_H, BOARD_W is the top right corner

Placement = np.ndarray
# dims = [BOARD_H, BOARD_W], dtype = np.uint8
# all values are zero except that corresponding to the marble insertion point
# the nonzero value is either WHITE_TURN or BLACK_TURN according to the color
# of marble to be placed

Rotation = np.ndarray
# dims = [2, 2], dtype = np.uint8
# all values are zero except that corresponding to the rotated quadrant
# the nonzero value is 1 for clockwise rotation and -1 for counterclockwise

Move = typing.Tuple[Placement, Rotation]


class Game():
    def __init__(self):
        self.board: Board = _generate_clean_board()
        self.turn = WHITE_TURN

    def make_move(self, move: Move):
        self.board = _apply_move(self.board, move)
        self.turn *= -1


def _generate_clean_board():
    return np.zeros(BOARD_DIMS, dtype=np.int8)


def _can_place_mask(board):
    return board == 0


def _apply_move(board, move: Move):
    placement, rotation = move
    board = board + placement
    _apply_rotation(board, rotation)
    return board


@numba.jit()
def _apply_rotation(board, rotation):
    q_i = 0
    q_j = 0
    direction = 0
    for i in range(2):
        for j in range(2):
            if rotation[i, j] != 0:
                q_i = i
                q_j = j
                direction = rotation[i, j]

    middles = np.array(((0, 1), (1, 0), (2, 1), (1, 2)))
    corners = np.array(((0, 0), (2, 0), (2, 2), (0, 2)))
    _make_swaps(board, middles, direction)
    _make_swaps(board, corners, direction)


@numba.jit()
def _make_swaps(board, swaps, direction):
    for i in range(2):
        from_i = swaps[i * 2][0]
        from_j = swaps[i * 2][1]
        to_i = swaps[(i * 2 + direction) % 4][0]
        to_j = swaps[(i * 2 + direction) % 4][1]

        temp = board[to_i, to_j]
        board[to_i, to_j] = board[from_i, from_j]
        board[from_i, from_j] = temp
