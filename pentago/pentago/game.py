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

N_IN_A_ROW = 5

Board = np.ndarray
# dims = [BOARD_H, BOARD_W], dtype = np.uint8
# values of 0 represent a blank space
# nonzero values are either WHITE_TURN or BLACK_TURN,
# representing the respective color of marble
# index 0, 0 is the bottom left corner
# index BOARD_H, BOARD_W is the top right corner

Placement = np.ndarray
# dims = [BOARD_H, BOARD_W], dtype = np.bool
# all values are False except that corresponding to the marble insertion point
# of marble to be placed

Rotation = np.ndarray
# dims = [2, 2], dtype = np.uint8
# all values are zero except that corresponding to the rotated quadrant
# the nonzero value is 1 for clockwise rotation and -1 for counterclockwise


class Game():
    """A nice wrapper class for game related functionality.

    All real logic is in module methods.

    """
    def __init__(self):
        self.board: Board = generate_clean_board()
        self.turn = WHITE_TURN

    def make_move(self, placement, rotation):
        self.board = apply_move(self.board, self.turn, placement, rotation)
        self.turn *= -1

    def check_game_over(self) -> typing.Optional[int]:
        """Returns WHITE_TURN or BLACK_TURN if the respective player has won, 0 if the
        game is a draw, and None if the game is not over.

        """
        return check_game_over(self.board)


@numba.jit(numba.int8[:, :]())
def generate_clean_board():
    return np.zeros((BOARD_H, BOARD_W), dtype=np.int8)


@numba.jit(numba.boolean[:, :](numba.int8[:, :],))
def can_place_mask(board):
    return board == 0


@numba.jit(nopython=True)
def _make_swaps(board, q_i, q_j, swaps, direction):
    for i in range(2):
        from_i = swaps[i * 2][0] + q_i * 3
        from_j = swaps[i * 2][1] + q_j * 3
        to_i = swaps[(i * 2 + direction) % 4][0] + q_i * 3
        to_j = swaps[(i * 2 + direction) % 4][1] + q_j * 3

        temp = board[to_i, to_j]
        board[to_i, to_j] = board[from_i, from_j]
        board[from_i, from_j] = temp


@numba.jit((numba.int8[:, :], numba.int8[:, :]), nopython=True)
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

    middles = np.array(((1, 0), (2, 1), (1, 2), (0, 1)))
    corners = np.array(((0, 0), (2, 0), (2, 2), (0, 2)))
    _make_swaps(board, q_i, q_j, middles, direction)
    _make_swaps(board, q_i, q_j, corners, direction)


@numba.jit(numba.int8[:, :](numba.int8[:, :],
                            numba.int8,
                            numba.boolean[:, :],
                            numba.int8[:, :]),
           nopython=True)
def apply_move(board, turn, placement, rotation):
    board = board + placement.astype(np.int8) * turn
    _apply_rotation(board, rotation)
    return board


@numba.jit()
def _check_line(board, i, j, i_stride, j_stride):
    sign = 0
    while i < BOARD_H and j < BOARD_W:
        if board[i, j] != sign:
            run = 0
            sign = board[i, j]
        run += 1
        if run == 5 and sign != 0:
            return sign
        i += i_stride
        j += j_stride
    return 0


@numba.jit(numba.optional(numba.int32)(numba.int8[:, :],), nopython=True)
def check_victory(board):
    # check rows
    for i in range(BOARD_H):
        r = _check_line(board, i, 0, 0, 1)
        if r != 0:
            return r

    # check columns
    for j in range(BOARD_W):
        r = _check_line(board, 0, j, 1, 0)
        if r != 0:
            return r

    # check diagonals
    diagonals = [_check_line(board, 0, 0, 1, 1),
                 _check_line(board, 1, 0, 1, 1),
                 _check_line(board, 0, 1, 1, 1),
                 _check_line(board, 5, 0, -1, 1),
                 _check_line(board, 4, 0, -1, 1),
                 _check_line(board, 5, 1, -1, 1)]
    for i in range(6):
        if diagonals[i] != 0:
            return diagonals[i]


@numba.jit(numba.boolean(numba.int8[:, :],), nopython=True)
def check_draw(board):
    return np.all(board)


@numba.jit(numba.optional(numba.int32)(numba.int8[:, :],), nopython=True)
def check_game_over(board):
    victory = check_victory(board)
    draw = check_draw(board)
    if victory is not None:
        return victory
    elif draw:
        return 0
    else:
        return None