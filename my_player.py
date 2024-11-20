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
        #convert the layout to a tuple for hashing!
        return layout

    def isNext(self, state0, state1):
        env0 = self.getLayout(state0)
        env1 = self.getLayout(state1)

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
                # Check if the state is in memory
                if self.getLayout(new_state) in self.memory:
                    value = self.memory[self.getLayout(new_state)]
                else:
                    value, _ = self.min_value(new_state, alpha, beta, maxDepth)
                    self.memory[self.getLayout(new_state)] = value
                
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
                # Check if the state is in memory
                if self.getLayout(new_state) in self.memory:
                    value = self.memory[self.getLayout(new_state)]
                else:
                    value, _ = self.max_value(new_state, alpha, beta, maxDepth)
                    self.memory[self.getLayout(new_state)] = value

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
                maxDepth = currentStep + 7
            
            case _:
                maxDepth = 40

        _, best_action = self.alphaBetaSearch(current_state, self.MIN, self.MAX, maxDepth)

        return best_action
    

'''
2. Iterative Deepening:
Iterative deepening combines the benefits of depth-first search and breadth-first search. It performs a series of depth-limited searches, gradually increasing the depth limit until the time runs out.

3. Transposition Tables:
Use a transposition table (hash table) to store the evaluation of previously visited game states. This avoids redundant calculations and speeds up the search.

4. Move Ordering:
Improve the efficiency of alpha-beta pruning by ordering moves so that the best moves are evaluated first. This increases the likelihood of pruning branches early.

5. Heuristic Evaluation Function:
Enhance your heuristic evaluation function to more accurately assess the value of game states. A better heuristic leads to better decision-making by the algorithm.

6. Quiescence Search:
Extend the search at "quiet" positions to avoid the horizon effect, where the algorithm misses important tactical moves just beyond the search depth.

7. Parallel Processing:
Leverage parallel processing to evaluate multiple branches of the search tree simultaneously. This can significantly speed up the search on multi-core processors.

8. Opening Book:
Use an opening book to store known good moves for the initial stages of the game. This can save time by avoiding unnecessary search in well-known positions.

Implementation of Iterative Deepening:
def iterativeDeepening(state, maxDepth):
    bestMove = None
    for depth in range(1, maxDepth + 1):
        bestMove = alphaBetaSearch(state, depth, float('-inf'), float('inf'), True)
    return bestMove

Implementation of Transposition Tables:
transposition_table = {}

def alphaBetaSearch(state, depth, alpha, beta, maximizingPlayer):
    state_hash = hash(state)
    if state_hash in transposition_table:
        return transposition_table[state_hash]

    if depth == 0 or state.is_terminal():
        eval = heuristic_evaluation(state)
        transposition_table[state_hash] = eval
        return eval

    if maximizingPlayer:
        maxEval = float('-inf')
        for child in state.get_children():
            eval = alphaBetaSearch(child, depth - 1, alpha, beta, False)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        transposition_table[state_hash] = maxEval
        return maxEval
    else:
        minEval = float('inf')
        for child in state.get_children():
            eval = alphaBetaSearch(child, depth - 1, alpha, beta, True)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        transposition_table[state_hash] = minEval
        return minEval

Implementation of Move Ordering:
def order_moves(moves):
    # Implement a heuristic to order moves
    return sorted(moves, key=lambda move: heuristic_evaluation(move.get_next_state()), reverse=True)

def alphaBetaSearch(state, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or state.is_terminal():
        return heuristic_evaluation(state)

    if maximizingPlayer:
        maxEval = float('-inf')
        for child in order_moves(state.get_children()):
            eval = alphaBetaSearch(child, depth - 1, alpha, beta, False)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = float('inf')
        for child in order_moves(state.get_children()):
            eval = alphaBetaSearch(child, depth - 1, alpha, beta, True)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval




'''