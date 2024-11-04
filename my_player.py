from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import numpy as np

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
        self._table = dict()

    def alphaBetaSearch(self, state: GameState, alpha: float, beta: float):
        value,move = self.max_value(state, alpha, beta)
        return (value, move)
    
    def max_value(self, state: GameState, alpha: float, beta: float):
        if state.is_done():
            return state.get_player_score(self), None
        bestValue = self.MIN
        bestAction = None
        for action in state.generate_possible_heavy_actions():
            new_state = action.get_next_game_state()

            # Save time by checking if the state is in the table
            if new_state in self._table:
                value = self._table[new_state]
            else:
                value, _ = self.min_value(new_state, alpha, beta)
                self._table[new_state] = value
            
            if value > bestValue:
                bestValue = value
                bestAction = action
                alpha = max(alpha, bestValue)
            if bestValue >= beta:
                return (bestValue, bestAction)
        return (bestValue, bestAction)
    
    def min_value(self, state: GameState, alpha:float, beta):
        if state.is_done():
            return state.get_player_score(self), None
        bestValue = self.MAX
        bestAction = None
        for action in state.generate_possible_heavy_actions():
            new_state = action.get_next_game_state()

            # Save time by checking if the state is in the table
            if new_state in self._table:
                value = self._table[new_state]
            else:
                value, _ = self.max_value(new_state, alpha, beta)
                self._table[new_state] = value

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
        
        best_score, best_action = self.alphaBetaSearch(current_state, self.MIN, self.MAX)

        return best_action