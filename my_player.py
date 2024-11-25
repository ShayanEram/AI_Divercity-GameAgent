from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError
# Custom imports
import numpy as np
from seahorse.game.light_action import LightAction
from game_state_divercite import GameStateDivercite

class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that makes random moves.

    Attributes:
        piece_type (str): piece type of the player
    """
    MIN = -np.inf
    MAX = np.inf

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerDivercite instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name)
        self.memory = dict()
        self.memory_limit = 5000

    def getScore(self, state: GameState):
        opponentId = [player.get_id() for player in state.players if player.get_id() != self.get_id()][0]
        score = state.scores[self.get_id()] - state.scores[opponentId]
        
        # if diversityState.get_player_id() == 1: # if enemy gets a diversity, punish minimax
        #     if diversityState.check_divercite():
        #         score -= 3
        
        # # if two cities can get a point from one ressorce, reward minimax
        # positions = state.get_player_position(self)
        # # if diversityState.get_neighbours(positions):
        #     # if positions[0] + 1 and positions[1] + 1 == ?
        return score

    def getLayout(self, state: GameState):
        layout = state.get_rep().get_env()
        return layout
    
    def addMemory(self, key, value):
        if len(self.memory) > self.memory_limit:
            self.memory.popitem()  # Remove the oldest item
        compressed_key = self.compress_state(key)
        self.memory[compressed_key] = value
    
    def getMemory(self, key):
        compressed_key = self.compress_state(key)
        return self.memory.get(compressed_key)

    def compress_state(self, state_dict):
        layout_str = str(sorted(state_dict.items()))
        compressed = []
        count = 1
        prev_char = layout_str[0]

        for char in layout_str[1:]:
            if char == prev_char:
                count += 1
            else:
                compressed.append(f"{count}{prev_char}")
                prev_char = char
                count = 1

        compressed.append(f"{count}{prev_char}")
        return ''.join(compressed)

    def decompress_state(self, compressed_str):
        decompressed = []
        count = ''
        
        for char in compressed_str:
            if char.isdigit():
                count += char
            else:
                decompressed.extend([char] * int(count))
                count = ''
        
        layout_str = ''.join(decompressed)
        items = eval(layout_str)
        return dict(items)

    def isNext(self, state0: GameState, state1: GameState):
        env0 = state0.get_rep().get_env()
        env1 = state1.get_rep().get_env()

        pos0 = set(env0.keys())
        pos1 = set(env1.keys())

        x , y = list((pos1 - pos0))[0]

        for i in [-1,1]:
            if (x+i,y) in pos0:
                return True
            if (x,y+i) in pos0:
                return True
        
        return False

    def alphaBetaSearch(self, state: GameState, alpha: float, beta: float, maxDepth: int):
        value,move = self.max_value(state, alpha, beta, maxDepth)
        return (value, move)
    
    def max_value(self, state: GameState, alpha: float, beta: float, maxDepth: int):
        if state.is_done() or state.get_step() == maxDepth:
            return (self.getScore(state), None)
        bestValue = self.MIN
        bestAction = None
        for action in state.generate_possible_heavy_actions():
            new_state = action.get_next_game_state()

            # Check if the next state is next to the current state
            passed = False
            if state.get_step() < 25:
                if self.isNext(state, new_state):
                    passed = True
            else:
                passed = True

            if passed:
                layout = self.getLayout(new_state)
                value = self.getMemory(layout)
                if value is None:
                    value, _ = self.min_value(new_state, alpha, beta, maxDepth)
                    self.addMemory(layout, value)
                
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                    alpha = max(alpha, bestValue)
                if bestValue >= beta:
                    return (bestValue, bestAction)

        return (bestValue, bestAction)
    
    def min_value(self, state: GameState, alpha:float, beta, maxDepth: int):
        if state.is_done() or state.get_step() == maxDepth:
            return (self.getScore(state), None)
        bestValue = self.MAX
        bestAction = None
        for action in state.generate_possible_heavy_actions():
            new_state = action.get_next_game_state()

            # Check if the next state is next to the current state
            passed = False
            if state.get_step() < 25:
                if self.isNext(state, new_state):
                    passed = True
            else:
                passed = True

            if passed:
                layout = self.getLayout(new_state)
                value = self.getMemory(layout)
                if value is None:
                    value, _ = self.max_value(new_state, alpha, beta, maxDepth)
                    self.addMemory(layout, value)

                if value < bestValue:
                    bestValue = value
                    bestAction = action
                    beta = min(beta, bestValue)
                if bestValue <= alpha:
                    return (bestValue, bestAction)
                
        return (bestValue, bestAction)

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        currentStep = current_state.get_step()
        self.memory = dict()

        match currentStep:
            case 0:
                return LightAction({"piece":'RC', "position": (5, 4)})
            
            case _ if 0 < currentStep < 15:
                maxDepth = currentStep + 5
            
            case _ if 15 <= currentStep < 20:
                maxDepth = currentStep + 6
            
            case _ if 20 <= currentStep < 30:
                maxDepth = currentStep + 6
            
            case _:
                maxDepth = 40

        _, best_action = self.alphaBetaSearch(current_state, self.MIN, self.MAX, maxDepth)

        return best_action
    
