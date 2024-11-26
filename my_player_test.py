from player_divercite import PlayerDivercite
from board_divercite import BoardDivercite
from seahorse.game.light_action import LightAction
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
        self._next_table_reset = 1

    def getScore0(self, state: GameState):
        return state.scores[self.get_id()] - state.scores[self.opponentId]

    def isWin(self, state: GameState):
        score = self.getScore0(state)
        if score == 0:
            return score
        else:
            return score/abs(score)

    def getNDivercite(self, state: GameState):
        env = state.get_rep().get_env()
        board = BoardDivercite(
            env=env, dim=state.get_rep().get_dimensions())
        d = board.get_dimensions()

        n_divercite = sum([state.check_divercite((i, j), board=board) for i in range(d[0]) for j in range(d[1])
                           if state.in_board((i, j)) and env.get((i, j)) and env.get((i, j)).get_type()[1] == 'C' and env[(i, j)].get_owner_id() == self.get_id()])
        return n_divercite

    def getDivercitePieces(self, state: GameState):

        pieces_left = state.players_pieces_left[self.get_id()]
        divercite_pieces = 0
        for piece in pieces_left:
            if piece[1] == 'R' and pieces_left[piece] > 0:
                divercite_pieces += 1

        coeff = max(1 - self.current_step/20, 0)

        return divercite_pieces * coeff

    def maxValue(self, state: GameState, alpha: float, beta: float, max_depth: int):
        if state.is_done() or state.step == max_depth:
            if max_depth == 40:
                return (self.getScore0(state), None)
            else:
                return (self.getScore0(state) + self.getDivercitePieces(state), None)

        best_score = -np.inf
        best_action = None
        possible_actions = state.generate_possible_heavy_actions()
        for action in possible_actions:
            next_state = action.get_next_game_state()

            if next_state in self._table:
                score = self._table[next_state]

            else:
                score, _ = self.minValue(next_state, alpha, beta, max_depth)
                self._table[next_state] = score

            if score > best_score:
                best_score = score
                best_action = action
                alpha = max(alpha, best_score)
            if best_score >= beta:
                return (best_score, best_action)
        return (best_score, best_action)

    def minValue(self, state: GameState, alpha: float, beta: float, max_depth: int):

        if state.is_done() or state.step == max_depth:
            if max_depth == 40:
                return (self.getScore0(state), None)
            else:
                return (self.getScore0(state) + self.getDivercitePieces(state), None)

        best_score = np.inf
        best_action = None
        possible_actions = state.generate_possible_heavy_actions()
        for action in possible_actions:
            next_state = action.get_next_game_state()

            if next_state in self._table:
                score = self._table[next_state]

            else:
                score, _ = self.maxValue(next_state, alpha, beta, max_depth)
                self._table[next_state] = score

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

        self.opponentId = [player.get_id()
                           for player in current_state.players if player.get_id() != self.get_id()][0]

        threshold = 30
        exploration_depth = 4
        current_step = current_state.step
        self.current_step = current_step

        if current_step == 0:
            data = {"piece": 'RC', "position": (5, 4)}
            action = LightAction(data)
            return (action)

        elif current_step < 20:
            max_depth = current_step + 4

            self._table = dict()
            self._next_table_reset = current_step + 2 + \
                exploration_depth - exploration_depth % 2

        elif current_step < threshold:
            max_depth = current_step + exploration_depth

            if current_step == self._next_table_reset:

                self._table = dict()
                self._next_table_reset = max_depth - \
                    (max_depth-current_step) % 2

        else:
            if current_step == threshold or current_step == threshold + 1:
                self._table = dict()
            max_depth = 40

        best_score, best_action = self.maxValue(current_state,
                                                alpha=-np.inf,
                                                beta=np.inf,
                                                max_depth=max_depth)

        return best_action

        # TODO
        raise MethodNotImplementedError()
