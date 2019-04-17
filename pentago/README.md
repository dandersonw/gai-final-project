It is intended that you install this package in your python virtualenv/Conda environment.

You can play a console game with `play_console.py`:

``` shell
python play_console.py --black-agent random --white-agent human
```

You can download a pretrained (but not very good) model from [my personal website](http://derickanderson.xyz/files/trained_pentago_model.pkl).
For completeness, I note that you shouldn't run untrusted Python pickle files.
You can play against that model with:

``` shell
python play_console.py --black-agent neural --black-pickle trained_pentago_model.pkl
```

Agents are specified with strings and configs.
Available agents are:

* random: Takes random legal moves
* minimax: Moves based on minimax search
* human: passes control to a human on the console
* neural: AlphaZero-alike, MCTS + resnet

Two humans can play each other at one time.
It is also possible to pit two AI against each other.

Config is passed as a JSON string.
Available options depend on the agent.

* minimax
  * depth: the depth of search
  * evaluation_function: the evaluation function to use
* neural: many parameters, but the one you might change on the command line:
  * mcts_simulations: how many rollouts of Monte-Carlo tree search to perform

There is a harness for pitting agents against each other:

``` shell
(gai-project) 鳳 python tally_agent_stats.py --first-agent neural --first-pickle ~/Downloads/trained_pentago_model.pkl --second-agent random --trials 100
first/second/Draw: 99/0/0: 100%|███████████████████████████████████████| 100/100 [20:43<00:00,  9.53s/it]
First wins: 100, Second wins: 0, Draws: 0
```
