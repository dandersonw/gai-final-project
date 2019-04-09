import random
import json
from os import path

from . import game

zobrist_keys = {'black': [], 'white': [], 'black_turn': ''}
keyfile_path = 'zobrist_keys.txt'

saved_boards = {}

# indices of each quadrant, clockwise from bot-left
rotation_indices = [0, 6, 12, 13, 14, 8, 2, 1]

# 30 31 32 33 34 35
# 24 25 26 27 28 29
# 18 19 20 21 22 23
# 12 13 14 15 16 17
#  6  7  8  9 10 11
#  0  1  2  3  4  5


def pad_key(key):
    return '{0:0{1}x}'.format(key, 16)


def xor(key_a, key_b):
    result = int(key_a, 16) ^ int(key_b, 16)
    return pad_key(result)


# encodes a board state into a Zobrist hash
def encode_board(board_state, turn):
    result = pad_key(0)
    boardstring = ""
    for i in range(game.BOARD_H):
        for j in range(game.BOARD_W):
            if board_state[i][j] == -1:
                # Black token in square
                result = xor(result, zobrist_keys['black'][(i * 6) + j])
                boardstring += '@'
            elif board_state[i][j] == 1:
                # White token in square
                result = xor(result, zobrist_keys['white'][(i * 6) + j])
                boardstring += 'O'
            else:
                # No token in square
                boardstring += '.'

    if turn < 1:  # Black to move
        result = xor(result, zobrist_keys['black_turn'])

    # Store boardstate in memory
    saved_boards[result] = {}
    saved_boards[result]['board'] = boardstring
    return result


# decodes a board state from a previously saved Zobrist hash
def decode_board(z_hash):
    try:
        boardstring = saved_boards[z_hash]
        board = game.generate_clean_board
        index = 0
        for c in boardstring:
            if c == 'O':
                board[int(index / 6)][index % 6] = 1
            if c == '@':
                board[int(index / 6)][index % 6] = -1
            index += 1

    except KeyError:
        return None


# progresses a Zobrist Key using the given move
def progress_key(z_hash, turn, placement, rotation):
    result = z_hash
    boardstring = saved_boards[z_hash]

    index = 0
    # get move position
    for i in range(game.BOARD_H):
        for j in range(game.BOARD_W):
            if placement[i][j]:
                index = (i * 6) + j

    # Add token to board
    try:
        if turn == 1:
            result = xor(result, zobrist_keys['white'][index])
        else:
            result = xor(result, zobrist_keys['black'][index])
    except IndexError:
        print("\n" + str(index) + "\n")

    boardstring[index] = 'O' if turn == 1 else '@'
    indices = [8]

    # Rotate the board!
    for i in range(2):
        for j in range(2):
            if rotation[i, j] != 0:
                indices = [x + (i * 18) + (j * 3) for x in rotation_indices]
                direction = rotation[i, j]

    for x in range(8):
        if boardstring[indices[x]] == 'O':
            result = xor(result, zobrist_keys['white'][indices[x]])                          # Remove old token
            result = xor(result, zobrist_keys['white'][indices[(x + (2 * direction)) % 8]])  # add rotated token
        elif boardstring[indices[x]] == '@':
            result = xor(result, zobrist_keys['black'][indices[x]])                          # Remove old token
            result = xor(result, zobrist_keys['black'][indices[(x + (2 * direction)) % 8]])  # add rotated token
    result = xor(result, zobrist_keys['black_turn'])
    return result


# Loads keys from the keyfile, or generates keys if a valid set doesn't exist
def load_keys():
    if path.isfile(keyfile_path):
        with open(keyfile_path) as keyfile:
            new_keys = json.load(keyfile)
            keyfile.close()

        # validation for key file
        try:
            zobrist_keys['black'] = [new_keys['black'][x] for x in range(36)]
            zobrist_keys['white'] = [new_keys['white'][x] for x in range(36)]
            zobrist_keys['black_turn'] = new_keys['black_turn']

        except (json.JSONDecodeError, IndexError, KeyError):
            print("Improper key formatting, generating new set...")
            generate_keys()
            load_keys()

    else:
        print("No keys found, generating new set...")
        generate_keys()
        load_keys()


def generate_keys():
    zobrist_keys['black'] = [pad_key(random.getrandbits(64)) for x in range(36)]
    zobrist_keys['white'] = [pad_key(random.getrandbits(64)) for x in range(36)]
    zobrist_keys['black_turn'] = pad_key(random.getrandbits(64))
    # Generates a random 64-length bitstring for each unique combo of piece and location
    # These keys get XOR'ed together to generate a bit hash for the board state.

    # Reset saved boards - previous hashes are useless
    global saved_boards
    saved_boards = {}

    with open(keyfile_path, 'w+') as keyfile:
        json.dump(zobrist_keys, keyfile)
        keyfile.close()
    return zobrist_keys


load_keys()
