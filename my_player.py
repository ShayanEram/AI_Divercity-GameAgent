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
    MIN = -np.inf
    MAX = np.inf

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self.memory = dict()
        self.memory_limit = 40000

    # General Functions ...................................................................................................
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

    def getLayout(self, state: GameState):
        layout = state.get_rep().get_env()
        return layout
    
    # def sort_moves(self, state: GameState, actions) -> list:
    #     scored_moves = [(action, self.heuristic_evaluation(state, action)) for action in actions]
    #     scored_moves.sort(key=lambda x: x[1], reverse=True)  # Sort moves in descending order of their scores
    #     sorted_actions = [action for action, score in scored_moves]
    #     return sorted_actions
    
    def get_opponent_id(self, state: GameState):
        return [player.get_id() for player in state.players if player.get_id() != self.get_id()][0]

    # def get_adjacent_resources(self, pos, state: GameState):
    #     x, y = pos
    #     adjacent_positions = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    #     return [state.get_rep().get_env().get(p) for p in adjacent_positions if p in state.get_rep().get_env()]

    # def get_adjacent_cities(self, pos, state: GameState):
    #     x, y = pos
    #     adjacent_positions = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    #     return [p for p in adjacent_positions if state.get_rep().get_env().get(p) == 'city']
    
    # Memory functions .............................................................................................
    def addMemory(self, key, value):
        if len(self.memory) > self.memory_limit:
            (k := next(iter(self.memory))), self.memory.pop(k)
        compressed_key = self.compress_state(key)
        self.memory[compressed_key] = value
    
    def getMemory(self, key):
        compressed_key = self.compress_state(key)
        return self.memory.get(compressed_key)

    def compress_state(self, state_dict):
        layout_str = str(sorted(state_dict.items()))
        return layout_str
        # compressed = []
        # count = 1
        # prev_char = layout_str[0]

        # for char in layout_str[1:]:
        #     if char == prev_char:
        #         count += 1
        #     else:
        #         compressed.append(f"{count}{prev_char}")
        #         prev_char = char
        #         count = 1

        # compressed.append(f"{count}{prev_char}")
        # return ''.join(compressed)

    # def decompress_state(self, compressed_str):
    #     decompressed = []
    #     count = ''
        
    #     for char in compressed_str:
    #         if char.isdigit():
    #             count += char
    #         else:
    #             decompressed.extend([char] * int(count))
    #             count = ''
        
    #     layout_str = ''.join(decompressed)
    #     items = eval(layout_str)
    #     return dict(items)

    # Heuristic functions .........................................................................................
    def getScore(self, state: GameState):
        opponentId = self.get_opponent_id(state)
        return state.scores[self.get_id()] - state.scores[opponentId]
        # return self.combined_heuristic(state, self.get_id())

    # def heuristic_evaluation(self, state: GameState, action) -> float:
    #     new_state = action.get_next_game_state()
    #     score = self.getScore(new_state)
    #     return score

    # def diversite_bonus_heuristic(self, state: GameState, player_id: str):
    #     score = 0
    #     for pos, piece in state.get_rep().get_env().items():
    #         if piece == 'city' and state.get_rep().get_player(pos) == player_id:
    #             adjacent_resources = self.get_adjacent_resources(pos, state)
    #             if len(set(adjacent_resources)) == 4:
    #                 score += 5  # Diversité bonus points
    #     return score

    # def resource_color_matching_heuristic(self, state: GameState, player_id: str):
    #     score = 0
    #     for pos, piece in state.get_rep().get_env().items():
    #         if piece == 'city' and state.get_rep().get_player(pos) == player_id:
    #             city_color = state.get_rep().get_city_color(pos)
    #             adjacent_resources = self.get_adjacent_resources(pos, state)
    #             matching_resources = [r for r in adjacent_resources if r == city_color]
    #             score += len(matching_resources)
    #     return score

    # def resource_placement_advantage_heuristic(self, state: GameState, player_id: str):
    #     score = 0
    #     for pos, piece in state.get_rep().get_env().items():
    #         if piece == 'resource' and state.get_rep().get_player(pos) == player_id:
    #             resource_color = state.get_rep().get_resource_color(pos)
    #             adjacent_cities = self.get_adjacent_cities(pos, state)
    #             for city_pos in adjacent_cities:
    #                 city_owner = state.get_rep().get_player(city_pos)
    #                 if city_owner == player_id:
    #                     if resource_color == state.get_rep().get_city_color(city_pos):
    #                         score += 1
    #                 else:
    #                     if resource_color == state.get_rep().get_city_color(city_pos):
    #                         score -= 1  # Penalize if the resource helps the opponent
    #     return score
    
    # def central_control_heuristic(self, state: GameState, player_id: str):
    #     central_positions = [(4, 4), (4, 5), (5, 4), (5, 5)]  # Example central positions
    #     score = 0
    #     for pos in central_positions:
    #         if state.get_rep().get_env().get(pos) == player_id:
    #             score += 2  # Prioritize central control
    #     return score

    # def opponent_disruption_heuristic(self, state: GameState, player_id: str):
    #     score = 0
    #     opponent_id = self.get_opponent_id()
    #     for pos, piece in state.get_rep().get_env().items():
    #         if piece == 'city' and state.get_rep().get_player(pos) == opponent_id:
    #             adjacent_resources = self.get_adjacent_resources(pos, state)
    #             if len(set(adjacent_resources)) == 3:  # Opponent close to Diversité
    #                 score += 3  # Disrupt by placing a fourth resource not forming Diversité
    #     return score
    
    # def combined_heuristic(self, state: GameState, player_id: str):
    #     diversite_score = self.diversite_bonus_heuristic(state, player_id)
    #     color_matching_score = self.resource_color_matching_heuristic(state, player_id)
    #     placement_advantage_score = self.resource_placement_advantage_heuristic(state, player_id)
    #     central_control_score = self.central_control_heuristic(state, player_id)
    #     disruption_score = self.opponent_disruption_heuristic(state, player_id)

    #     # Combine scores with weights
    #     combined_score = (
    #         1.5 * diversite_score +
    #         1.0 * color_matching_score +
    #         0.8 * placement_advantage_score +
    #         1.0 * central_control_score +
    #         1.2 * disruption_score
    #     )
    #     return combined_score

    # Alpha-Beta Search .............................................................................................
    def alphaBetaSearch(self, state: GameState, alpha: float, beta: float, maxDepth: int):
        value,move = self.max_value(state, alpha, beta, maxDepth)
        return (value, move)
    
    def max_value(self, state: GameState, alpha: float, beta: float, maxDepth: int):
        if state.is_done() or state.get_step() == maxDepth:
            return (self.getScore(state), None)
        bestValue = self.MIN
        bestAction = None
        actions = state.generate_possible_heavy_actions()
        # sorted_actions = self.sort_moves(state, actions)  # Sort the moves
        for action in actions:
            new_state = action.get_next_game_state()

            if self.isNext(state, new_state) or state.get_step() >= 25:
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
        actions = state.generate_possible_heavy_actions()
        # sorted_actions = self.sort_moves(state, actions)  # Sort the moves
        for action in actions:
            new_state = action.get_next_game_state()

            if self.isNext(state, new_state) or state.get_step() >= 25:
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

    # Main function ...............................................................................................
    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
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