import numpy as np
from collections import defaultdict
import os

class Nim_Model:
    def __init__(self, action_size, lr=0.01):
        super(Nim_Model, self).__init__()
        self.action_size = action_size
        self.policy_head = defaultdict(lambda: np.ones(action_size) / action_size)
        self.value_head = defaultdict(float)

        self.lr = lr

    def predict(self, state):
        if len(state.shape) != 1:
            raise Exception('predict function only processes individual state')
        
        state = state.tobytes()

        policy = self.policy_head[state]
        value = self.value_head[state]

        return policy, value
    
    def inference(self, state):
        if len(state.shape) != 1:
            raise Exception('predict function only processes individual state')
        
        state = state.tobytes()

        if state in self.policy_head:
            policy = self.policy_head[state]
            value = self.value_head[state]
        else:
            policy = np.ones(self.action_size) / self.action_size
            value = 0

        return policy, value
    

    def update(self, state, target_pi, target_v):
        state = state.tobytes()

        policy = self.policy_head[state]
        value = self.value_head[state]
        
        # Update with moving average
        self.policy_head[state] = (1 - self.lr) * policy + self.lr * target_pi
        self.policy_head[state] /= self.policy_head[state].sum() 

        self.value_head[state] = float(np.clip((1 - self.lr) * value + self.lr * target_v, -1, 1))



if __name__ == '__main__':
    from NimEnvironments import NimEnv

    game = NimEnv(num_piles=5)
    state = game.reset()
    model = Nim_Model(action_size=game.action_size)
    pred = model.predict(state)
    print(pred)
