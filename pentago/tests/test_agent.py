import pytest
import json

import pentago
import pentago.agent
import pentago.search_ai


def test_get_agent_for_key():
    assert pentago.get_agent_for_key(None, 'human') is None
    assert type(pentago.get_agent_for_key(None, 'random')) == pentago.agent.RandomAgent
    assert type(pentago.get_agent_for_key(None, 'minimax')) == pentago.search_ai.MinimaxSearchAgent

    with pytest.raises(NameError):
        pentago.get_agent_for_key(None, 'foo')

    agent = pentago.search_ai.MinimaxSearchAgent(depth=3)
    config = json.loads(json.dumps(agent.to_params()))
    assert pentago.get_agent_for_key(None, 'minimax', config).depth == 3
