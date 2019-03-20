import numpy as np

import pentago
import pentago.game


def test_move():
    game = pentago.Game()
    assert game.turn == 1
    placement = np.zeros_like(game.board, dtype=np.bool)
    placement[0, 0] = 1
    rotation = np.asarray([[1, 0],
                           [0, 0]],
                          dtype=np.int8)
    game.make_move(placement, rotation)
    expected_board = np.zeros_like(game.board)
    expected_board[2, 0] = 1
    assert np.all(game.board == expected_board)

    placement = np.zeros_like(game.board, dtype=np.bool)
    placement[5, 5] = 1
    rotation = np.asarray([[0, 0],
                           [0, -1]],
                          dtype=np.int8)
    game.make_move(placement, rotation)
    expected_board[5, 3] = -1
    assert np.all(game.board == expected_board)

    placement = np.zeros_like(game.board, dtype=np.bool)
    placement[1, 3] = 1
    rotation = np.asarray([[0, 1],
                           [0, 0]],
                          dtype=np.int8)
    game.make_move(placement, rotation)
    expected_board[2, 4] = 1
    assert np.all(game.board == expected_board)

    placement = np.zeros_like(game.board, dtype=np.bool)
    placement[3, 2] = 1
    rotation = np.asarray([[0, 0],
                           [-1, 0]],
                          dtype=np.int8)
    game.make_move(placement, rotation)
    expected_board[3, 0] = -1
    assert np.all(game.board == expected_board)


def test_check_game_over():
    assert pentago.game.check_game_over(pentago.game.generate_clean_board()) is None

    board = pentago.game.generate_clean_board()
    board[0, 0] = 1
    board[0, 1] = 1
    board[0, 2] = 1
    board[0, 3] = 1
    board[0, 4] = 1
    assert pentago.game.check_game_over(board) == 1

    board = -1 * board
    assert pentago.game.check_game_over(board) == -1

    board = pentago.game.generate_clean_board()
    board[0, 0] = 1
    board[0, 1] = 1
    board[0, 2] = 1
    board[0, 3] = 1
    assert pentago.game.check_game_over(board) is None

    board = pentago.game.generate_clean_board()
    board[0, 0] = 1
    board[1, 1] = 1
    board[2, 2] = 1
    board[3, 3] = 1
    board[4, 4] = 1
    assert pentago.game.check_game_over(board) == 1

    board = np.asarray([[ 1,  1,  1, -1,  1,  1],  # noqa
                        [ 1,  1,  1, -1,  1,  1],  # noqa
                        [ 1,  1,  1, -1,  1, -1],  # noqa
                        [-1, -1, -1,  1, -1,  1],  # noqa
                        [ 1,  1, -1,  1, -1,  1],  # noqa
                        [ 1,  1,  1, -1,  1,  1]],   # noqa
                       dtype=np.int8)
    assert pentago.game.check_game_over(board) == 0


def test_proposal():
    board = np.asarray([[ 0,  0,  0,  0,  0,  0], # noqa
                        [ 1, -1,  0,  1,  0,  0],  # noqa
                        [ 0,  0,  0, -1,  1,  0],  # noqa
                        [-1,  0,  0,  0, -1,  0],  # noqa
                        [ 0,  1, -1,  1,  1, -1],  # noqa
                        [ 1,  0,  0,  0,  0,  0]],  # noqa
                       dtype=np.int8)
    placement = np.zeros_like(board, dtype=np.bool)
    placement[2, 1] = True
    rotation = np.asarray([[-1, 0],
                           [0, 0]],
                          dtype=np.int8)
    board = pentago.game.apply_move(board, -1, placement, rotation)
    expected_board = np.asarray([[ 0,  1,  0,  0,  0,  0], # noqa
                                 [ 0, -1, -1,  1,  0,  0],  # noqa
                                 [ 0,  0,  0, -1,  1,  0],  # noqa
                                 [-1,  0,  0,  0, -1,  0],  # noqa
                                 [ 0,  1, -1,  1,  1, -1],  # noqa
                                 [ 1,  0,  0,  0,  0,  0]],  # noqa
                                dtype=np.int8)
    assert np.all(expected_board == board)
