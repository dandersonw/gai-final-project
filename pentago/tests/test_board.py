import numpy as np

import pentago


def test_move():
    game = pentago.Game()
    assert game.turn == 1
    placement = np.zeros_like(game.board)
    placement[0, 0] = 1
    rotation = np.asarray([[1, 0],
                           [0, 0]],
                          dtype=np.int8)
    game.make_move((placement, rotation))
    expected_board = np.zeros_like(game.board)
    expected_board[2, 0] = 1
    assert np.all(game.board == expected_board)
