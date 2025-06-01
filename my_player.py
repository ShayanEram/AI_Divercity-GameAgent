"""
@file my_player.py
@brief This is the core implementation of the minimax agent
@author Raphael Tournier, Shayan Eram
"""
from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError

# Added Imports
from seahorse.game.light_action import LightAction
import itertools
import hashlib

# Added Imports (Need to be added in requirements.txt)
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
        # Add any information you want to store about the player here
        # self.json_additional_info = {}
        self.MAX = np.inf
        self.MIN = -np.inf
        self.memory = {}
        self.MEMORY_LIMIT = 500000
        self.BASE_DEPTH = 4
        self.color = 'B'
        self.positions = set()
        self.currentStep = 0

    # Evaluation Functions------------------------------------------------------------------------------
    def endSearch(self, state: GameState, depth: int) -> bool:
        if state.is_done() or state.step == depth:
            return True
        else:
            return False

    def getLayout(self, state: GameState) -> dict:
        return state.get_rep().get_env()

    def getStep(self, state: GameState) -> int:
        return state.step
    
    def getOpponentId(self, state: GameState) -> int:
        return [player.get_id() for player in state.get_players() if player.get_id() != self.get_id()][0]
    
    def playerPiecesLeft(self, state: GameState) -> dict:
        piecesLeft = state.players_pieces_left[self.get_id()]
        return piecesLeft
    
    def getPositionCoordinates(self, state1: GameState, state2: GameState) -> tuple:
        env1 = self.getLayout(state1)
        env2 = self.getLayout(state2)

        pos1 = set(env1.keys())
        pos2 = set(env2.keys())

        new_positions = pos2 - pos1

        new_x, new_y = next(iter(new_positions))
        return (new_x, new_y)
    
    # Heuristic Functions-------------------------------------------------------------------------------
    def getColorPenalty(self, state: GameState) -> int:
        penalty = 0
        board = self.getLayout(state)

        for pos, piece in board.items():
            if piece.get_type()[2] == self.color and piece.get_type()[1] == 'C':
                x, y = pos
                neighbors = state.get_rep().get_neighbours(x, y)
                color_counts = {
                    color: sum(
                        1 for neighbor in neighbors.values()
                        if neighbor[0] != 'EMPTY' and neighbor[0].get_type()[0] == color
                    )
                    for color in {'R', 'G', 'B', 'Y'}
                }
                penalty += sum(max(0, count - 1) for count in color_counts.values())

        return penalty        

    def getCityDistances(self, state: GameState) -> float:
        board = self.getLayout(state)
        positions = [np.array(pos) for pos in board if board[pos].get_type()[2] == self.color and board[pos].get_type()[1] == 'C']

        sumDistances = sum(np.linalg.norm(p1 - p2) for p1, p2 in itertools.combinations(positions, 2))
        
        return sumDistances

    def getResourcePieceCount(self, state: GameState) -> int:
        piecesLeft = self.playerPiecesLeft(state)
        piecesCount = sum(1 for piece, count in piecesLeft.items() if piece[1] == 'R' and count > 0)
        
        return piecesCount

    def getScoreDifference(self, state: GameState) -> int:
        selfScore = state.scores[self.get_id()]
        opponentScore = state.scores[self.getOpponentId(state)]
        deltaScore = selfScore - opponentScore
        
        return deltaScore        

    def getScore(self, state: GameState) -> (float | int):
        
        currentStep = self.currentStep
        
        if currentStep < 16 : 
            return self.getScoreDifference(state) - self.getColorPenalty(state) - self.getCityDistances(state)
        elif currentStep < 25 : 
            return self.getScoreDifference(state) - self.getColorPenalty(state)
        elif currentStep < 30 :
            return self.getScoreDifference(state) + self.getResourcePieceCount(state)
        else:
            return self.getScoreDifference(state) 
    
    def isValidState(self, state1: GameState, state2: GameState) -> bool:
        
        if self.getStep(state2) > 30:
            return True
        
        if self.getStep(state1) < 4 and self.getStep(state1) == self.currentStep:            
            
            x, y = self.getPositionCoordinates(state1, state2)
            if (x,y) in [(3,4),(4,5),(5,4),(4,3)]:
                return True   
            else:
                return False  
        
        if (self.getStep(state1) - self.currentStep) % 2 == 0 and self.getStep(state1) < 16:
            remainingPieces = self.playerPiecesLeft(state2)
            totalRessources = 0
            for piece in remainingPieces:
                if piece[1] == 'R':
                    totalRessources += remainingPieces[piece]
            if totalRessources < 12:
                return False

        x, y = self.getPositionCoordinates(state1, state2)
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            if (x + dx, y + dy) in self.positions:
                return True

        return False

    # Memory Functions---------------------------------------------------------------------------------
    def addMemory(self, key, value) -> None:
        if len(self.memory) >= self.MEMORY_LIMIT:
            (k := next(iter(self.memory))), self.memory.pop(k)
        self.memory[key] = value

    def getMemory(self, key) -> (float | None):
        return self.memory.get(key)
    
    def getHashLayout(self, state: GameState) -> str:
        return hashlib.sha256((str(state.get_rep()) + str(state.players_pieces_left[self.get_id()])).encode()).hexdigest()

    def checkMemory(self, state: GameState) -> tuple:
        layout = self.getHashLayout(state)
        value = self.getMemory(layout)
        return (layout, value)

    # Alpha-Beta Pruning---------------------------------------------------------------------------------
    def alphaBetaSearch(self, state: GameState, alpha: float, beta: float, maxDepth: int) -> tuple:
        value, action = self.maxValue(state, alpha, beta, maxDepth)
        return (value, action)

    def maxValue(self, state: GameState, alpha: float, beta: float, maxDepth: int) -> tuple:
        if self.endSearch(state, maxDepth):
            return (self.getScore(state), None)
        bestValue = self.MIN
        bestAction = None
        actions = state.generate_possible_heavy_actions()
        for action in actions:
            new_state = action.get_next_game_state()

            if self.isValidState(state, new_state):
                layout, value = self.checkMemory(new_state)
                if value is None:
                    value, _ = self.minValue(new_state, alpha, beta, maxDepth)
                    self.addMemory(layout, value) 

                if value > bestValue:
                    bestValue = value
                    bestAction = action
                    alpha = max(alpha, bestValue)
                if bestValue >= beta:
                    return (bestValue, bestAction)
        return (bestValue, bestAction)

    def minValue(self, state: GameState, alpha: int, beta: int, maxDepth: int) -> tuple:
        if self.endSearch(state, maxDepth):
            return (self.getScore(state), None)
        bestValue = self.MAX
        bestAction = None
        actions = state.generate_possible_heavy_actions()
        for action in actions:
            new_state = action.get_next_game_state()

            if self.isValidState(state, new_state):
                layout, value = self.checkMemory(new_state)
                if value is None:
                    value, _ = self.maxValue(new_state, alpha, beta, maxDepth)
                    self.addMemory(layout, value)

                if value < bestValue:
                    bestValue = value
                    bestAction = action
                    beta = min(beta, bestValue)
                if bestValue <= alpha:
                    return (bestValue, bestAction)
        return (bestValue, bestAction)

    #-------------------------------------------------------------------------------------------------
    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        currentStep = self.getStep(current_state)
        self.currentStep = currentStep
        self.positions = set(self.getLayout(current_state).keys())

        if currentStep == 0:
            action = LightAction({"piece":'RC', "position": (5, 4)})
            self.color = 'W'
            return action
        
        if currentStep < 24:
            self.memory = {}
            maxDepth = currentStep + self.BASE_DEPTH
        
        elif currentStep < 30:
            self.memory = {}
            maxDepth = currentStep + self.BASE_DEPTH + 1
        
        elif currentStep < 32:
            self.memory = {}
            maxDepth = 40
        
        else:
            maxDepth = 40

        if remaining_time < 500 and currentStep < 32:
            maxDepth = self.BASE_DEPTH - 1

        _, action = self.alphaBetaSearch(current_state, self.MIN, self.MAX, maxDepth)

        return action