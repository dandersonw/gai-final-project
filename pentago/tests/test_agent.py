import pytest
import json

import pentago
import pentago.agent
import pentago.minimax
import pentago.learning_agent


def test_get_agent_for_key():
    assert pentago.get_agent_for_key(None, 'human') is None
    assert type(pentago.get_agent_for_key(None, 'random')) == pentago.agent.RandomAgent
    assert type(pentago.get_agent_for_key(None, 'minimax')) == pentago.minimax.MinimaxSearchAgent

    with pytest.raises(NameError):
        pentago.get_agent_for_key(None, 'foo')

    agent = pentago.minimax.MinimaxSearchAgent(depth=3)
    config = json.loads(json.dumps(agent.to_params()))
    assert pentago.get_agent_for_key(None, 'minimax', config).depth == 3

    agent = pentago.get_agent_for_key(None, 'neural')
    config = json.loads(json.dumps(agent.to_params()))
    pentago.get_agent_for_key(None, 'neural', config)
