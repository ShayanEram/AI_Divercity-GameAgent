from player_divercite import PlayerDivercite
from board_divercite import BoardDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.game.light_action import LightAction
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import numpy as np


class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that makes random moves.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerDivercite instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name)
        self.memory_limit = 500000
        self.memory = dict()



# Memory........................................................................................................................

    def addMemory(self, key, value):
        if len(self.memory) > self.memory_limit:
            (k:=next(iter(self.memory)), self.memory.pop(k))  
        self.memory[key] = value
        
    def getMemory(self, key):
        self.memoryCount += 1

        # score = self.memory.pop(key)
        # self.memory[key] = score
        # return score
    
        return self.memory.get(key)


    def getLayout(self, state: GameState, toString = False):
        if toString:            
            return hash(str(state.get_rep()))
        else:
            return state.get_rep().get_env()        
    
    # Score...............................................................

    def getScore0(self, state: GameState):

        opponentId = [player.get_id()
                      for player in state.players if player.get_id() != self.get_id()][0]

        return state.scores[self.get_id()] - state.scores[opponentId]
    
    # Valid States.................................................................................
    
    def isValid(self, state1, state2):

        if state2.step > 30:
            return True

        env1 = state1.get_rep().get_env()
        env2 = state2.get_rep().get_env()

        pos2 = set(env2.keys())
        pos1 = set(env1.keys())

        x , y = list((pos2 - pos1))[0]

        for (i,j) in [(i,j) for i in [-1,0,1] for j in [-1,1,0]]:
            if (x+i,y+j) in self._positions:
                return True
        
        return False
                 

    def maxValue(self, state: GameState, alpha: float, beta: float, max_depth: int):
        if state.is_done() or state.step == max_depth:
            return (self.getScore0(state), None)
        best_score = -np.inf
        best_action = None
        possible_actions = state.generate_possible_heavy_actions()
        for action in possible_actions:

            next_state = action.get_next_game_state()
    
            if self.isValid(state, next_state):

                layout = self.getLayout(next_state, toString = True)
                if layout in self.memory:
                    score = self.getMemory(layout)
                else:

                    score, _ = self.minValue(next_state, alpha, beta, max_depth)
                    self.addMemory(layout, score)                

                if score > best_score:
                    best_score = score
                    best_action = action
                    alpha = max(alpha, best_score)
                if best_score >= beta:
                    return (best_score, best_action)                

                
        return (best_score, best_action)

    def minValue(self, state: GameState, alpha: float, beta: float, max_depth: int):
        if state.is_done() or state.step == max_depth:
            return (self.getScore0(state), None)
        best_score = np.inf
        best_action = None
        possible_actions = state.generate_possible_heavy_actions()
        for action in possible_actions:

            next_state = action.get_next_game_state()
            
            if self.isValid(state, next_state):

                layout = self.getLayout(next_state, toString = True)
                if layout in self.memory:
                    score = self.getMemory(layout)
                else:
                

                    score, _ = self.maxValue(next_state, alpha, beta, max_depth)
                    self.addMemory(layout, score)         
                
                if score < best_score:
                    best_score = score
                    best_action = action
                    beta = min(beta, best_score)
                if best_score <= alpha:
                    return (best_score, best_action)
                
                
        return (best_score, best_action)

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        self.memoryCount = 0
        
        current_step = current_state.step
        self._positions = set(current_state.get_rep().get_env().keys())

        if current_step == 0:
            data = {"piece": 'RC', "position": (5, 4)}
            action = LightAction(data)
            return (action)


        self.memory = dict()
        max_depth = current_step + 4

        _, best_action = self.maxValue(current_state,
                                                alpha=-np.inf,
                                                beta=np.inf,
                                                max_depth=max_depth)

        
        # print(self.memoryCount)
        
        return best_action
