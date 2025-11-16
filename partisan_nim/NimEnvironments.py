import numpy as np


class NimEnv:
    def __init__(self, num_piles=3):
        super(NimEnv, self).__init__()
        self.num_piles = num_piles
        

        self.evens = [x for x in range(self.num_piles) if x % 2 == 0]
        self.odds  = [x for x in range(self.num_piles) if x % 2 != 0]

        self.nim_unitary = NimUnitary(num_piles)
        self.all_actions = self.complete_legal_actions()
        self.board_size = self.nim_unitary.board_size
        self.action_size = len(self.all_actions)

        self.player = 1
        

    def set_board(self, state):
        state = state.copy()
        self.nim_unitary.board = state
    
    def get_observation(self):
        observation = self.nim_unitary.observe()
        return observation.copy()
    
    def to_play(self):
        return self.player

    def reset(self):
        self.player = 1
        self.nim_unitary._initialize_board()
        return self.get_observation()
    
    def step(self, action):
        action = self.all_actions[action]
        self.nim_unitary.action(action)
        
        player_done = self.nim_unitary.is_player_done(self.player)
        # game_done = self.nim_unitary.is_game_done()

        reward = 0
        # if player_done:
        #     reward = 1
        if player_done:
            reward = -1

        obs = self.get_observation()
        
        self.player *= -1
        
        return obs.copy(), reward, player_done

    def sample_random_action(self):
        mask = np.array(self.get_action_masks())
        action = np.random.choice(np.where(mask == 1)[0])
        return action

    def get_action_masks(self):
        mask = [0.0 for _ in range(len(self.all_actions))]
        all_legal_action_indices = self.get_all_legal_action_indices()
        for idx in all_legal_action_indices:
            mask[idx] = 1.0
        return mask.copy()
    
    def get_all_legal_actions(self):
        legal_actions = []
        if self.player == 1:
            pile_ids = self.evens
        else:
            pile_ids = self.odds

        for pile_index in pile_ids:
            start_pile_index = pile_index ** 2 + pile_index
            end_pile_index = (pile_index + 1) ** 2 + pile_index

            pile = self.nim_unitary.board[start_pile_index: end_pile_index]
            num_matches = int(sum(pile))
            if num_matches > 0:
                for match in range(num_matches):
                    legal_actions.append((pile_index, match+1))
        return legal_actions.copy()


    def get_all_legal_action_indices(self):
        legal_action_indices = []
        for action in self.get_all_legal_actions():
            index = self.all_actions.index(action)
            legal_action_indices.append(index)
        return legal_action_indices.copy()

    
    def complete_legal_actions(self):
        legal_actions = []
        for pile_index in range(self.nim_unitary.num_piles):
            start_pile_index = pile_index ** 2 + pile_index
            end_pile_index = (pile_index + 1) ** 2 + pile_index

            pile = self.nim_unitary.board[start_pile_index: end_pile_index]
            num_matches = int(sum(pile))
            if num_matches > 0:
                for match in range(num_matches):
                    legal_actions.append((pile_index, match+1))
        return legal_actions.copy()


class NimUnitary(object):
    def __init__(self, num_piles=3):
        self.num_piles = num_piles
        self.num_matches = num_piles ** 2 
        self.board_size = (self.num_piles - 1) + self.num_matches
        self._initialize_board()
        self.rewards = {'WON': 1.0,
                        'VALID_ACTION': 0}
        self.evens = [x for x in range(self.num_piles) if x % 2 == 0]
        self.odds  = [x for x in range(self.num_piles) if x % 2 != 0]
        
    def _initialize_board(self):
        self.is_action_valid = True
        self.board = np.ones((self.board_size,), dtype=np.float64)
        for i in range(1, self.num_piles):
            self.board[i**2 + i-1] = -1  # noise used to separate piles
    
    def action(self, action):
        pile_index, match_num = action
        start_pile_index = pile_index**2 + pile_index
        end_pile_index = (pile_index + 1)**2 + pile_index
        pile = self.board[start_pile_index: end_pile_index]
        count = 1
        for i in range(len(pile)):
            if (pile[i] == 1.0) and (count <= match_num):
                pile[i] = 0.0
                count += 1
        self.board[start_pile_index: end_pile_index] = pile
        
    def evaluate(self):
        if self.is_done():
            return self.rewards['WON']
        else:
            return self.rewards['VALID_ACTION']

    def get_match_numbers(self):
        counts = []
        current = 0

        for x in self.board:
            if x == -1:
                counts.append(current)
                current = 0
            elif x == 1:
                current += 1

        counts.append(current)

        return counts

    def is_game_done(self):
        if self.board.sum() == (self.num_piles - 1) * -1.0:
            return True
        else:
            return False
    
    def is_player_done(self, player):
        match_numbers = self.get_match_numbers()
        if player == 1:
            if sum([match_numbers[i] for i in self.evens]) == 0:
                return True
        else:
            if sum([match_numbers[i] for i in self.odds]) == 0:
                return True
        return False

    
    def observe(self):
        return self.board.copy()


if __name__ == "__main__":
    import random

    nim = NimEnv(num_piles=3)
    state = nim.reset()

    done = False
    total_reward = 0

    while not done:
        actions = nim.get_all_legal_action_indices()   # list of legal actions
        action = random.choice(actions)         # pick a random action
        print(nim.player)
        next_state, reward, done = nim.step(action)
        total_reward += reward

        print(f"Action: {nim.all_actions[action]}, Next State: {next_state}, Reward: {reward}, Done: {done}")
  