import tensorflow as tf
import numpy as np

import abc
import random
import string
import typing

from . import agent, memory, neural_model, util


class SelfPlayAgent(agent.AIAgent):
    @abc.abstractmethod
    def fit(self, exps: typing.List[memory.Experience]):
        pass


class NeuralAgent(SelfPlayAgent):
    key = 'neural'

    def __init__(self, *, model_name, model_params, weights_path=None):
        self.model_name = model_name
        self.model_params = model_params
        self.model: tf.keras.Model \
            = neural_model.model_for_key(model_name)(model_params)
        if weights_path is not None:
            self.model.load_weights(weights_path)

    def fit(self, exps: typing.List[memory.Experience]):
        pass

    def _strategy(self, board):
        pass

    @classmethod
    def load_params(cls, config):
        return dict()

    def to_params(self):
        weights_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase)
                               for _ in range(10))
        weights_path = util.AI_RESOURCE_PATH / weights_name
        self.model.save_weights(weights_path)
        return {'model_name': self.model_name,
                'model_params': self.model_params,
                'weights_path': weights_path}


def experiences_to_feed_dict(exps: typing.List[memory.Experience]):
    boards = np.stack([neural_model.board_to_input(e.board * e.turn) for e in exps])
    values = np.stack([e.value for e in exps])
    policies = np.stack([e.explanation for e in exps])
    return {'board': boards,
            'value_head': values,
            'policy_head': policies}
