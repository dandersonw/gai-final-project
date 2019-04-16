It is intended that you install this package in your python virtualenv/Conda environment.

You can play a console game with `play_console.py`:

``` shell
python play_console.py --black-agent random --white-agent human
```

You can download a pretrained (but not very good) model from [http://derickanderson.xyz/files/trained_pentago_model.pkl](my personal website).
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

Two humans can play each other at one time.
It is also possible to pit two AI against each other.

Config is passed as a JSON string.
Available options depend on the agent.

* minimax
  * depth: the depth of search
  * evaluation_function: the evaluation function to use

There is a harness for pitting agents against each other:

``` shell
(gai-project) 鳳 python tally_agent_stats.py --black-agent minimax --white-agent minimax --black-config '{"depth": 1}' --white-config '{"depth": 2}'
100%|██████████████████████████████████████████████████████████████████| 100/100 [03:16<00:00,  1.90s/it]
White wins: 100, Black wins: 0, Draws: 0
```

