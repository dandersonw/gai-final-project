import tensorflow as tf
import numpy as np

import abc
import random
import string
import typing

from . import game, agent, memory, neural_model, util, monte_carlo


Policy = np.ndarray
Value = float


class SelfPlayAgent(agent.AIAgent):
    @abc.abstractmethod
    def fit(self, exps: typing.List[memory.Experience]):
        pass

    @abc.abstractmethod
    def predict(self, board) -> typing.Tuple[Policy, Value]:
        pass


class NeuralAgent(SelfPlayAgent):
    key = 'neural'

    def __init__(self, *, model_name, model_params, weights_path=None):
        self.model_name = model_name
        self.model_params = model_params
        self.model: tf.keras.Model \
            = neural_model.model_for_key(model_name)(**model_params)
        if weights_path is not None:
            self.model.load_weights(weights_path)

    def fit(self, exps: typing.List[memory.Experience]):
        x, y = experiences_to_fit_data(exps)
        self.model.fit(x=x, y=y, batch_size=256, epochs=5)

    def predict(self, board):
        logits, value = self.model.predict(board[None, :, :, None].astype(np.float32))
        logits = np.squeeze(logits, 0)
        value = np.squeeze(value, 0)
        probs = game.masked_softmax(board, logits)
        return probs, value

    def _strategy(self, board):
        pi = monte_carlo.tree_search(board, self, temperature=0.5)
        idx = np.random.choice(game.PROBABILITY_OUT_DIM, p=pi)
        return game.move_from_flat_idx(idx), pi

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


def experiences_to_fit_data(exps: typing.List[memory.Experience]):
    boards = np.stack([(e.board * e.turn)[:, :, None].astype(np.float32) for e in exps])
    values = np.stack([e.value for e in exps])
    policies = np.stack([e.explanation for e in exps])
    x = {'board': boards}
    y = {'value_head': values,
         'policy_head': policies}
    return x, y
