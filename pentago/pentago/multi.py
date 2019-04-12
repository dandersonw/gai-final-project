import tensorflow as tf

import itertools
import time
import ray

from collections import Counter

from . import agent, memory, controller, evaluate


workers = {}


def self_play_games(agent: agent.Agent,
                    view: memory.MemoryView,
                    num_games,
                    num_workers=2,
                    verbose=False):
    """Using a worker pool, play agent against itself and add those experiences to
    view.

    """
    agent_key = agent.key
    if agent_key == 'neural':
        agent_params = agent.to_params(in_memory_weights=True)
    else:
        agent_params = agent.to_params()
    start_time = time.time()
    workers = _get_workers('self_play', num_workers)

    for worker in workers:
        worker.set_agent.remote(agent_key, agent_params)

    game_tasks = []
    for i in range(num_games):
        game_tasks.append(workers[i % num_workers].run_one_game.remote())

    game_results = list(itertools.chain.from_iterable(ray.get(game_tasks)))
    view._add_experiences(game_results)
    end_time = time.time()
    time_taken = end_time - start_time
    if verbose:
        print('{} seconds taken, {} per game'
              .format(time_taken, time_taken / num_games))


def _get_workers(key, number):
    existing = workers.get(key, [])
    for i in range(number - len(existing)):
        existing.append(worker_classes[key].remote())
    workers[key] = existing
    return existing[:number]


def tally_wins(agents,
               trials,
               num_workers=2):
    agent_specs = []
    for agentt in agents:
        if agentt.key == 'neural':
            params = agentt.to_params(in_memory_weights=True)
        else:
            params = agentt.to_params()
        agent_specs.append([agentt.key, params])

    workers = _get_workers('tally_wins', num_workers)

    for worker in workers:
        worker.set_agents.remote(agent_specs)

    tasks = [workers[t % num_workers].run_one_trial.remote(t) for t in range(trials)]

    wins = Counter()
    for result in ray.get(tasks):
        wins.update(result)

    return wins


@ray.remote
class SelfPlayWorker():
    def __init__(self):
        self.agent = None

    def set_agent(self, agent_key, agent_params):
        if self.agent is not None:
            if self.agent.key == 'neural':
                del self.agent.model
            del self.agent
            tf.keras.backend.clear_session()
        self.agent = agent.get_agent_for_key(None,
                                             agent_key,
                                             agent_params)

    def run_one_game(self):
        view = memory.MemoryView()
        cont = controller.Controller([self.agent] * 2, view)
        cont.play_game()
        return view.get_experiences()


@ray.remote
class TallyWinWorker():
    def __init__(self):
        self.agents = None

    def set_agents(self, agent_specs):
        if self.agents is not None:
            for agentt in self.agents:
                if agentt.key == 'neural':
                    del agentt.model
                del agentt
            tf.keras.backend.clear_session()
        self.agents = []
        for agent_key, agent_params in agent_specs:
            self.agents.append(agent.get_agent_for_key(None,
                                                       agent_key,
                                                       agent_params))

    def run_one_trial(self, t):
        result = {0: 0, -1: 0, 1: 0}
        evaluate._run_one_trial(self.agents,
                                None,
                                t,
                                result,
                                False)
        return result


worker_classes = {'self_play': SelfPlayWorker,
                  'tally_wins': TallyWinWorker}
