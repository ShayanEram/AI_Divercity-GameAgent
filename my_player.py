from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import numpy as np

from seahorse.game.light_action import LightAction

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

    def alphaBetaSearch(self, state: GameState, alpha: float, beta: float, maxDepth: int):
        value,move = self.max_value(state, alpha, beta, maxDepth)
        return (value, move)
    
    def max_value(self, state: GameState, alpha: float, beta: float, maxDepth: int):
        if state.is_done() or state.get_step() == maxDepth:
            return (state.get_player_score(self), None)
        bestValue = self.MIN
        bestAction = None
        for action in state.generate_possible_heavy_actions():
            new_state = action.get_next_game_state()
            value, _ = self.min_value(new_state, alpha, beta, maxDepth)
            if value > bestValue:
                bestValue = value
                bestAction = action
                alpha = max(alpha, bestValue)
            if bestValue >= beta:
                return (bestValue, bestAction)
        return (bestValue, bestAction)
    
    def min_value(self, state: GameState, alpha:float, beta, maxDepth: int):
        if state.is_done() or state.get_step() == maxDepth:
            return (state.get_player_score(self), None)
        bestValue = self.MAX
        bestAction = None
        for action in state.generate_possible_heavy_actions():
            new_state = action.get_next_game_state()
            value, _ = self.max_value(new_state, alpha, beta, maxDepth)
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
        maxDepth = currentStep + 4

        # This gets 1 or 2 more point for x4 the time!
        # match currentStep:
        #     case 0:
        #         return LightAction({"piece":'RC', "position": (5, 4)})
            
        #     case _ if 0 < currentStep < 15:
        #         maxDepth = currentStep + 4
            
        #     case _ if 15 <= currentStep < 20:
        #         maxDepth = currentStep + 5
            
        #     case _ if 20 <= currentStep < 30:
        #         maxDepth = currentStep + 6
            
        #     # case _ if 30 <= currentStep < 40:
        #     #     maxDepth = currentStep + 15
            
        #     case _:
        #         maxDepth = 50
        

        _, best_action = self.alphaBetaSearch(current_state, self.MIN, self.MAX, maxDepth)

        return best_action