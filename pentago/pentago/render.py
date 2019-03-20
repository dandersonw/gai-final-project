import os

from . import game


WHITE_CHAR = 'O'
BLACK_CHAR = '@'
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
        line = [c for c in ' '.join(line)]
        line[game.QUADRANT_W * 2 - 1] = '|'
        lines.append(''.join(line))
    separator_line = ('-' * (game.QUADRANT_W * 2 - 1) +
                      '+' +
                      '-' * (game.QUADRANT_W * 2 - 1))
    lines.insert(game.QUADRANT_H, separator_line)
    lines = reversed(lines)
    return os.linesep.join(lines)
