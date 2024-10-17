from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError

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

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        
        # Heuristic evaluation functions
        def populateMyCity():
            pass

        def checkOpponentDiversity():
            pass

        def checkOpponentSingleScore():
            pass

        def blockOpponentDiversity():
            pass

        def blockOpponentSingleScore():
            pass

        def checkOppnentReserveresources():
            pass 
        
        # Minimax algorithm
        MIN= -1000000000
        MAX = 1000000000

        def alphaBetaSearch(state: GameState, alpha, beta):
            value,move = max_value(state, alpha, beta)
            return value, move
        
        def max_value(state: GameState, alpha, beta):
            if state.is_done():
                return state.get_player_score(self.piece_type), None
            bestValue = MIN
            bestMove = None
            for action in state.get_possible_heavy_actions():
                new_state = action.get_next_game_state()
                value, _ = min_value(new_state, alpha, beta)
                if value > bestValue:
                    bestValue = value
                    bestMove = action
                    alpha = max(alpha, bestValue)
                if bestValue >= beta:
                    return bestValue, bestMove
            return bestValue, bestMove
        
        def min_value(state: GameState, alpha, beta):
            if state.is_done():
                return state.get_player_score(self.piece_type), None
            bestValue = MAX
            bestMove = None
            for action in state.get_possible_heavy_actions():
                new_state = action.get_next_game_state()
                value, _ = max_value(new_state, alpha, beta)
                if value < bestValue:
                    bestValue = value
                    bestMove = action
                    beta = min(beta, bestValue)
                if bestValue <= alpha:
                    return bestValue, bestMove
            return bestValue, bestMove

        best_action = None
        best_score = MIN

        for action in current_state.get_possible_heavy_actions():
            new_state = action.get_next_game_state()
            score, _ = alphaBetaSearch(new_state, MIN, MAX)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action





   







