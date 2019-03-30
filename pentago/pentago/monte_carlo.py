import numpy as np

from collections import deque

from . import game, learning_agent


class Node():
    def __init__(self, board, turn):
        self.board = board
        self.turn = turn
        self.edges = None
        self.value = None
        self.edge_prior = None
        self.edge_visit_counts = None
        self.edge_evaluations = None
        self.edge_action_values = None
        self.game_over = game.check_game_over(self.board)

    def is_leaf(self):
        return self.edges is None or self.game_over is not None

    def expand(self, model, noise=None):
        assert self.is_leaf()

        if self.game_over is not None:
            self.value = self.game_over
            return

        self.edge_prior, self.value = model.predict(self.board * self.turn)
        if noise is not None:
            self.edge_prior += noise
        self.edges = [None] * self.edge_prior.shape[0]
        self.edge_visit_counts = np.zeros_like(self.edge_prior)
        self.edge_evaluations = np.zeros_like(self.edge_prior)
        self.edge_action_values = np.zeros_like(self.edge_prior)

    def get_edge(self, idx):
        if self.edges[idx] is None:
            move = game.move_from_flat_idx(idx)
            new_board = game.apply_move(self.board, self.turn, *move)
            new_turn = self.turn * -1
            new_node = Node(new_board, new_turn)
            edge = Edge(self, new_node, idx)
            self.edges[idx] = edge
        return self.edges[idx]

    def select_action(self):
        u = self.edge_prior / (1 + self.edge_visit_counts)
        idx = np.argmax(self.edge_action_values + u)
        return self.get_edge(idx)


class Edge():
    def __init__(self, parent, child, idx):
        self.parent = parent
        self.child = child
        self.idx = idx


def tree_search(board,
                model,
                temperature=1e-2,
                num_simulations=100):
    root = Node(board, 1)
    root.expand(model, noise=_get_root_noise())

    for i in range(num_simulations):
        path, end_node = _select_path(root)
        end_node.expand(model)
        _backprop(path, end_node.value)

    visit_counts = root.edge_visit_counts
    pi = ((visit_counts ** (1 / temperature))
          / (np.sum(visit_counts ** (1 / temperature))))
    return pi


def _select_path(root):
    node = root
    path = deque()
    while not node.is_leaf():
        edge = node.select_action()
        path.append(edge)
        node = edge.child
    return path, node


def _backprop(path, value):
    while path:
        edge = path.pop()
        p = edge.parent
        idx = edge.idx
        p.edge_visit_counts[idx] += 1
        p.edge_evaluations[idx] += value
        p.edge_action_values = (p.edge_evaluations[idx]
                                / p.edge_visit_counts[idx])


def _get_root_noise():
    noise = np.random.dirichlet(np.full([game.PROBABILITY_OUT_DIM], 0.05))
    return noise - 1 / game.PROBABILITY_OUT_DIM