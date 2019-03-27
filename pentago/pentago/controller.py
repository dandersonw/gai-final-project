import typing

from . import agent, view, game


class Controller():
    def __init__(self, agents: typing.List[agent.Agent],
                 view: typing.Optional[view.View]):
        self.agents = agents
        self.view = view

    def play_game(self):
        model = game.Game()
        winner = None
        while winner is None:
            for agentt in self.agents:
                if self.view is not None:
                    self.view.render(model)
                agentt.make_move(model)
                winner = model.check_game_over()
                if winner is not None:
                    break
        if self.view is not None:
            self.view.game_ended(winner)
        return winner
