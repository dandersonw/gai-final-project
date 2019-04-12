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

    @abc.abstractmethod
    def set_evaluation_mode(self, evaluation_mode):
        pass


class NeuralAgent(SelfPlayAgent):
    key = 'neural'

    def __init__(self, *,
                 model_name='residual_conv_net',
                 model_params=neural_model.DEFAULT_MODEL_PARAMS,
                 mcts_simulations=400,
                 weights_path=None):
        self.model_name = model_name
        self.model_params = model_params
        self.mcts_simulations = mcts_simulations
        self.evaluation_mode = False
        self.model: tf.keras.Model \
            = neural_model.model_for_key(model_name)(**model_params)
        if weights_path is not None:
            self.model.load_weights(weights_path)

    def fit(self,
            exps: typing.List[memory.Experience],
            augment=True,
            **kwargs):
        x, y = experiences_to_fit_data(exps, augment=augment)
        callbacks = [tf.keras.callbacks.EarlyStopping(mode='min',
                                                      patience=10,
                                                      restore_best_weights=True)]
        fit_kwargs = {'validation_split': 0.1,
                      'callbacks': callbacks,
                      'batch_size': 1024,
                      'verbose': 2,
                      'epochs': 100,
                      **kwargs}
        self.model.fit(x=x, y=y,
                       **fit_kwargs)

    def predict(self, board):
        logits, value = self.model.predict(board[None, :, :, None].astype(np.float32))
        logits = np.squeeze(logits, 0)
        value = np.squeeze(value, 0)
        probs = game.masked_softmax(board, logits)
        return probs, value

    def _strategy(self, board):
        temperature = .1 if self.evaluation_mode else 1

        pi = monte_carlo.tree_search(board,
                                     self,
                                     num_simulations=self.mcts_simulations,
                                     temperature=temperature)

        idx = np.random.choice(game.PROBABILITY_OUT_DIM, p=pi)
        return game.move_from_flat_idx(idx), pi

    def set_evaluation_mode(self, evaluation_mode):
        self.evaluation_mode = evaluation_mode

    @classmethod
    def load_params(cls, config):
        keys = {'model_name', 'model_params', 'weights_path', 'mcts_simulations'}
        keys = keys.intersection(config.keys())
        return {k: config[k] for k in keys}

    def to_params(self):
        weights_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase)
                               for _ in range(10))
        weights_path = str(util.AI_RESOURCE_PATH / weights_name)
        self.model.save_weights(weights_path)
        return {'model_name': self.model_name,
                'model_params': self.model_params,
                'weights_path': weights_path,
                'mcts_simulations': self.mcts_simulations}


def experiences_to_fit_data(exps: typing.List[memory.Experience], augment=True):
    boards = np.stack([e.board[:, :, None].astype(np.float32) for e in exps])
    values = np.stack([e.value for e in exps])
    policies = np.stack([e.explanation for e in exps])
    if augment:
        boards = np.concatenate([np.rot90(boards, axes=(1, 2), k=k)
                                 for k in range(4)],
                                axis=0)
        values = np.concatenate([values] * 4, axis=0)
        policies = np.concatenate([_rotate_policy(policies, k)
                                   for k in range(4)],
                                  axis=0)
    x = {'board': boards}
    y = {'value_head': values,
         'policy_head': policies}
    return x, y


def _rotate_policy(policy, k):
    result = np.reshape(policy, [-1, *game.PROBABILITY_OUT_SHAPE])
    result = np.rot90(result, k=k, axes=(1, 2))
    result = np.rot90(result, k=k, axes=(3, 4))
    return np.reshape(result, policy.shape)
