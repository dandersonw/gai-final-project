import numpy as np

import pentago
import os


def test_render_board():
    board = np.asarray([[ 0,  0,  0,  0,  0,  0], # noqa
                        [ 1, -1,  0,  1,  0,  0],  # noqa
                        [ 0,  0,  0, -1,  1,  0],  # noqa
                        [-1,  0,  0,  0, -1,  0],  # noqa
                        [ 0,  1, -1,  1,  1, -1],  # noqa
                        [ 1,  0,  0,  0,  0,  0]],  # noqa
                       dtype=np.int8)
    assert pentago.render_board(board) == os.linesep.join(['O . .|. . .',
                                                           '. O @|O O @',
                                                           '@ . .|. @ .',
                                                           '-----+-----',
                                                           '. . .|@ O .',
                                                           'O @ .|O . .',
                                                           '. . .|. . .'])
