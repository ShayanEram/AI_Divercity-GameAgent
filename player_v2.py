# Shayan Eram - 2084174 
# Raphaël Tournier – 2409579


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
        self.color = 'B'



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
            return hash(str(state.get_rep()) + str(state.players_pieces_left[self.get_id()]))
        else:
            return state.get_rep().get_env()        
    
    # Score...............................................................
            
    
    def getDivercitePieces(self, state: GameState):

        pieces_left = state.players_pieces_left[self.get_id()]
        divercite_pieces = 0
        for piece in pieces_left:
            if piece[1] == 'R':
                divercite_pieces += (pieces_left[piece] > 0)
        return divercite_pieces    
    
    def getDivercitePenalty(self, state: GameState):
        
        penalty = 0
        
        board = self.getLayout(state)
        for pos in board:
            if board[pos].get_type()[2] == self.color and board[pos].get_type()[1] == 'C':
                color = board[pos].get_type()[0]
                x,y = pos
                neighbors = state.get_rep().get_neighbours(x,y)
                colorCount = 0
                for k,v in neighbors.items():
                    if v[0] != 'EMPTY':
                        if v[0].get_type()[0] == color:
                            colorCount+=1
                penalty += max(0,colorCount-1)
        
        return penalty

                    

    def getDeltaScore(self, state: GameState):

        opponentId = [player.get_id()
                      for player in state.players if player.get_id() != self.get_id()][0]
        return state.scores[self.get_id()] - state.scores[opponentId]        
        
            

    def getScore(self, state):
        
        step = self.current_step
        
        if step < 25 : 
            return self.getDeltaScore(state) - self.getDivercitePenalty(state)
        elif step < 30 :
            return self.getDeltaScore(state) + self.getDivercitePieces(state)
        else:
            return self.getDeltaScore(state) 




    # Valid States.................................................................................
    
    def isValid(self, state1, state2):

        if state2.step > 30:
            return True
        
        if state1.step < 3 and state1.step == self.current_step:            
                
            env1 = self.getLayout(state1)
            env2 = self.getLayout(state2)

            pos2 = set(env2.keys())
            pos1 = set(env1.keys())

            x , y = list((pos2 - pos1))[0]
            
            if (x,y) in [(3,4),(4,5),(5,4),(4,3)]:
                return True   
            else:
                return False  
        
        if (state1.step - self.current_step) % 2 == 0 and state1.step < 15:
            pieces_left = state2.players_pieces_left[self.get_id()]
            totalRessources = 0
            for piece in pieces_left:
                if piece[1] == 'R':
                    totalRessources += pieces_left[piece]
            if totalRessources < 12:
                return False
            
        # if (state2.step < 30) and (state1.step - self.current_step) % 2 == 0:
        #     if self.getDivercitePieces(state2) < 4:
        #         return False


        env1 = state1.get_rep().get_env()
        env2 = state2.get_rep().get_env()

        pos2 = set(env2.keys())
        pos1 = set(env1.keys())

        x , y = list((pos2 - pos1))[0]

        for (i,j) in [(i,j) for i in [-1,0,1] for j in [-1,1,0]]:
            if (x+i,y+j) in self._positions:
                return True
        
        return False
                 
    # Minimax......................................................................................

    def maxValue(self, state: GameState, alpha: float, beta: float, max_depth: int):
        if state.is_done() or state.step == max_depth:
            return (self.getScore(state), None)
        best_score = -np.inf
        best_action = None
        
        state_step = state.step
        
        if state_step < 8 and state_step == self.current_step:
            possible_actions = list(action for action in state.generate_possible_heavy_actions())
            np.random.shuffle(possible_actions)
        else:
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
            return (self.getScore(state), None)
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
        self.current_step= current_step
        self._positions = set(current_state.get_rep().get_env().keys())

        if current_step == 0:
            data = {"piece": 'RC', "position": (5, 4)}
            action = LightAction(data)
            self.color = 'W'
            return (action)
            
        if current_step < 24  :
            self.memory = dict()
            max_depth = current_step + 4
        elif current_step < 30  :
            self.memory = dict()
            max_depth = current_step + 5
        elif current_step in {30,31}:
            self.memory = dict()
            max_depth = 40
        else:
            max_depth = 40


        best_score, best_action = self.maxValue(current_state,
                                                alpha=-np.inf,
                                                beta=np.inf,
                                                max_depth=max_depth)

        
        # print(self.memoryCount)
        
        return best_action
