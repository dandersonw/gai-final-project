import typing

from tqdm import tqdm

from . import agent, controller, learning_agent


def tally_wins(agents: typing.List[agent.Agent],
               verbose=False,
               trials=100):
    assert len(agents) == 2

    for agentt in agents:
        if isinstance(agentt, learning_agent.SelfPlayAgent):
            agentt.set_evaluation_mode(True)

    results = {0: 0, 1: 0, -1: 0}

    trials = list(range(trials))
    if verbose:
        with tqdm(trials) as trials:
            for t in trials:
                _run_one_trial(agents, trials, t, results, verbose)
    else:
        for t in trials:
            _run_one_trial(agents, trials, t, results, verbose)

    for agentt in agents:
        if isinstance(agentt, learning_agent.SelfPlayAgent):
            agentt.set_evaluation_mode(False)

    return results


def _run_one_trial(agents, trials, t, results, verbose):
    if verbose:
        trials.set_description('1/-1/D: {}/{}/{}'.format(results[1],
                                                         results[-1],
                                                         results[0]))
    parity = 1 if t % 2 == 0 else -1
    agents_ = agents if parity == 1 else list(reversed(agents))
    cont = controller.Controller(agents_, None)
    winner = cont.play_game() * parity
    results[winner] += 1