import numpy as np

import numba
import typing

from .import zobrist


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

PROBABILITY_OUT_SHAPE = (6, 6, 2, 2, 2)
PROBABILITY_OUT_DIM = 6 * 6 * 2 * 2 * 2


Board = np.ndarray
# dims = [BOARD_H, BOARD_W], dtype = np.uint8
# values of 0 represent a blank space
# nonzero values are either WHITE_TURN or BLACK_TURN,
# representing the respective color of marble
# index 0, 0 is the bottom left corner
# index BOARD_H, BOARD_W is the top right corner

Placement = np.ndarray
# dims = [BOARD_H, BOARD_W], dtype = np.int8
# all values are False except that corresponding to the marble insertion point
# of marble to be placed

Rotation = np.ndarray
# dims = [2, 2], dtype = np.uint8
# all values are zero except that corresponding to the rotated quadrant
# the nonzero value is 1 for clockwise rotation and -1 for counterclockwise

Move = typing.Tuple[Placement, Rotation]


class Game():
    """A nice wrapper class for game related functionality.

    All real logic is in module methods.

    """
    def __init__(self):
        self.board: Board = generate_clean_board()
        self.turn = WHITE_TURN
        self.key = zobrist.encode_board(self.board, WHITE_TURN)

    def make_move(self, move: Move):
        placement, rotation = move
        self.board = apply_move(self.board, self.turn, placement, rotation)
        self.turn *= -1

    def check_game_over(self) -> typing.Optional[int]:
        """Returns WHITE_TURN or BLACK_TURN if the respective player has won, 0 if the
        game is a draw, and None if the game is not over.

        """
        return check_game_over(self.board)

    def can_place_in_square(self, i, j) -> bool:
        return can_place_mask(self.board)[i, j]


@numba.jit(numba.int8[:, :](numba.int64, numba.int64),
           nopython=True,
           cache=True)
def generate_placement_from_indices(i, j):
    placement = np.zeros((BOARD_H, BOARD_W), dtype=np.int8)
    placement[i, j] = 1
    return placement


@numba.jit(numba.int8[:, :](numba.int32, numba.int32, numba.int8),
           nopython=True,
           cache=True)
def generate_rotation(q_i, q_j, direction):
    rotation = np.zeros((2, 2), dtype=np.int8)
    rotation[q_i, q_j] = direction
    return rotation


def flat_index_for_move(move):
    placement, rotation = move
    placement_idx = np.argmax(placement)
    rotation_idx = np.argmax(np.abs(rotation))
    return placement_idx * 8 + rotation_idx * 2 + (1 if np.ravel(rotation)[rotation_idx] == 1 else 0)


def move_from_flat_idx(idx):
    idx = np.unravel_index(idx, PROBABILITY_OUT_SHAPE)
    i, j, q_i, q_j, d = idx
    d = 1 if d == 1 else -1
    return (generate_placement_from_indices(i, j),
            generate_rotation(q_i, q_j, d))


@numba.jit(numba.int8[:, :](),
           nopython=True,
           cache=True)
def generate_clean_board():
    return np.zeros((BOARD_H, BOARD_W), dtype=np.int8)


@numba.jit(numba.boolean[:, :](numba.int8[:, :],),
           nopython=True,
           cache=True)
def can_place_mask(board):
    return board == 0


@numba.jit(nopython=True)
def mask_move_logits(board, flat_logits):
    for i in range(6):
        for j in range(6):
            if board[i, j] != 0:
                for k in range(8):
                    flat_logits[(i * 6 + j) * 8 + k] = -np.inf
    return flat_logits


@numba.jit(nopython=True)
def masked_softmax(board, flat_logits):
    mask_move_logits(board, flat_logits)
    exp = np.exp(flat_logits)
    return exp / np.sum(exp)


@numba.jit(nopython=True, cache=True)
def _make_swaps(board, q_i, q_j, locs, direction):
    vals = np.zeros((4,), dtype=np.int8)
    for i in range(4):
        vals[i] = board[q_i * 3 + locs[i][0], q_j * 3 + locs[i][1]]
    vals = np.roll(vals, direction)
    for i in range(4):
        board[q_i * 3 + locs[i][0], q_j * 3 + locs[i][1]] = vals[i]


@numba.jit((numba.int8[:, :], numba.int8[:, :]),
           nopython=True,
           cache=True)
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
                            numba.int8[:, :],
                            numba.int8[:, :]),
           nopython=True,
           cache=True)
def apply_move(board, turn, placement, rotation):
    # new_key, new_board = check_move(key, turn, placement, rotation)
    # if new_key:
    #     return new_board
    # else:
        # Board not explored, manually change and add to saved boards
    board = board + placement.astype(np.int8) * turn
    _apply_rotation(board, rotation)
    # zobrist.encode_board(board, turn * -1)
    return board


def check_move(key, turn, placement, rotation):
    try:
        # Check if board has already been seen in Zobrist hashes
        result = zobrist.decode_board(zobrist.progress_key(key, turn, placement, rotation))
        return True, result
    except KeyError:
        return False, None


@numba.jit(nopython=True)
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


@numba.jit(numba.optional(numba.int32)(numba.int8[:, :],),
           nopython=True,
           cache=True)
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


@numba.jit(numba.boolean(numba.int8[:, :],),
           nopython=True,
           cache=True)
def check_draw(board):
    return np.all(board)


@numba.jit(numba.optional(numba.int32)(numba.int8[:, :],),
           nopython=True,
           cache=True)
def check_game_over(board):
    victory = check_victory(board)
    draw = check_draw(board)
    if victory is not None:
        return victory
    elif draw:
        return 0
    else:
        return None
