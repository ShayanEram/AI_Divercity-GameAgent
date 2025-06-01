# Divercity_game_agent_AI

## Rules
**Divercité** is a strategic game where two players compete to accumulate the most points by placing their pieces (called cities) on a board, surrounded by resources of different colors. The aim of the game is to create configurations that maximize points according to several rules.

### Game Preparation
Players face each other, one playing with black pieces and the other with white pieces. Each player receives their cities and three resource tokens of each color.

### Game Play
Each turn, a player must place one of their pieces on an unoccupied square on the board. They have two options:
1. Place a city on a paved square.
2. Place a resource on a dotted square.
Once placed, the piece cannot be moved. Players alternate until all pieces are placed on the board. When all pieces have been played, one last resource square remains unoccupied, and the game is over.

### Scoring Points
Once all pieces are placed, players examine each of their cities, counting points based on the resources surrounding them. There are two scenarios for point calculation:
1. **Divercité**: If a city is surrounded by four different colored resources, it earns 5 points for its owner.
2. **Identical Resources**: If the city is not surrounded by four different resources, it earns 1 point for each adjacent resource of the same color as the city, earning between 1 and 4 points.

Resources placed on the board are neutral and benefit all surrounding cities, regardless of their color. For example, a player might place a red resource near their red cities to score points but must also consider that this action could benefit the opponent.

In case of a tie in score, players are differentiated based on their number of Divercités. If this is also tied, we count the number of cities surrounded by four resources of the same color, then those with three resources, and so on until a winner is determined. If a perfect tie occurs, the first player wins since the second player is considered to have an advantage.

## Instructions

The project is based on the open-source package seahorse. You can support it by giving a star on its GitHub page. To start a game, you must first install seahorse using the following command:
```bash
$ pip install seahorse colorama
```

Next, several execution modes are available via the presence of arguments. For example, `-r` allows you to save a game in a JSON file. To get the description of all the arguments, execute the following command:
```bash
$ python main_divercite.py -h
```

To start a local game with GUI, you can use the following command:
```bash
$ python main_divercite.py -t local random_player_divercite.py random_player_divercite.py
```

The following command starts a local game between the random agent and the greedy agent, logs are saved in a JSON, and the GUI is not opened:
```bash
$ python main_divercite.py -t local random_player_divercite.py greedy_player_divercite.py -r -g
```

If you want to organize a game against an agent from another group, you need to run the following command to host the match:
```bash
$ python main_divercite.py -t host_game -a <ip_address> random_player_divercite.py
```

The team you want to face should run:
```bash
$ python main_divercite.py -t connect -a <ip_address> random_player_divercite.py
```
You will need to replace `<ip_address>` with the IP address of the computer hosting the game. To find this, run the `ipconfig` (Windows) or `ifconfig` (Mac, Linux) command in a terminal.

To get familiar with the game initially, you can play manually against each other with the following command:
```bash
$ python main_divercite.py -t human_vs_human
```

Finally, to study the behavior of your agent, it might be interesting to play manually against it with:
```bash
$ python main_divercite.py -t human_vs_computer random_player_divercite.py
```

## Play
python main_divercite.py -t local my_player.py greedy_player_divercite.py -g (from command.txt)
