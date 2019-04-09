import pytest
import json
from os import path

import pentago
import pentago.zobrist as zb

hexdigits = "0123456789abcdefABCDEF"
test_keys = {}


def setup_module(module):
    print("setup_module         module: "+ module.__name__)
    global test_keys
    test_keys = zb.generate_keys()


def test_key_generation():
    # Check top-level categories
    assert 'white' in test_keys.keys()
    assert 'black' in test_keys.keys()
    assert 'black_turn' in test_keys.keys()

    # Check list lengths
    assert len(test_keys['white']) == 36
    assert len(test_keys['black']) == 36

    # Validate key values
    for key in [test_keys['white'][x] for x in range(36)]:
        assert type(key) == str
        assert len(key) == 16
        assert all(c in hexdigits for c in key)

    for key in [test_keys['black'][x] for x in range(36)]:
        assert type(key) == str
        assert len(key) == 16
        assert all(c in hexdigits for c in key)

    key = test_keys['black_turn']
    assert type(key) == str
    assert len(key) == 16
    assert all(c in hexdigits for c in key)

    # Validate written JSON file
    assert path.isfile(zb.keyfile_path)
    with open(zb.keyfile_path) as keyfile:
        stored_keys = json.load(keyfile)
        keyfile.close()

    assert all(stored_keys['white'][key] == test_keys['white'][key] for key in range(36))
    assert all(stored_keys['black'][key] == test_keys['black'][key] for key in range(36))
    assert stored_keys['black_turn'] == test_keys['black_turn']

