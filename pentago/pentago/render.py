import os

from . import game


WHITE_CHAR = 'O'
BLACK_CHAR = 'X'
BLANK_CHAR = '.'
CHAR_DICT = {0: BLANK_CHAR,
             game.WHITE_TURN: WHITE_CHAR,
             game.BLACK_TURN: BLACK_CHAR}


def render_board(board: game.Board):
    lines = []
    for i in range(game.BOARD_H):
        line = []
        for j in range(game.BOARD_W):
            line.append(CHAR_DICT[board[i, j]])
        lines.append(''.join(line))
    lines = reversed(lines)
    return os.linesep.join(lines)
