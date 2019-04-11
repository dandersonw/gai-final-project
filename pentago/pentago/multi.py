import itertools
import time
import ray

from . import agent, memory, controller


def self_play_games(agent: agent.Agent,
                    view: memory.MemoryView,
                    num_games,
                    num_workers=2,
                    verbose=False):
    """Using a worker pool, play agent against itself and add those experiences to
    view.

    """
    agent_key = agent.key
    agent_params = agent.to_params()
    start_time = time.time()
    workers = [SelfPlayWorker.remote(agent_key, agent_params)
               for _ in range(num_workers)]

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


@ray.remote
class SelfPlayWorker():
    def __init__(self, agent_key, agent_params):
        self.agent = agent.get_agent_for_key(None,
                                             agent_key,
                                             agent_params)

    def run_one_game(self):
        view = memory.MemoryView()
        cont = controller.Controller([self.agent] * 2, view)
        cont.play_game()
        return view.get_experiences()
