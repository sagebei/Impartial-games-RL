import numpy as np
from random import shuffle
from NimEnvironments import NimEnv
from monte_carlo_tree_search import MCTS
from ExpertPolicyValue import get_states_policies_values_masks
import ray
import copy


@ray.remote
class Simulation:
    def __init__(self, game, args):
        self.game = copy.deepcopy(game)
        self.args = args

    def execute_episode(self, model):
        self.model = model
        mcts = MCTS(self.game, self.model, self.args)
        train_examples = []
        
        state = self.game.reset()
        done = False
        n_moves = 0
        while not done:
            root = mcts.run(state, self.game.to_play(), is_train=True)
            action_probs = [0.0 for _ in range(self.game.action_size)]
            for action, child in root.children.items():
                action_probs[action] = child.visit_count
            action_probs = action_probs / np.sum(action_probs)

            train_examples.append((state, action_probs, self.game.to_play()))

            # Set the temperature for the first a few moves to 1, and the remaining moves to 0.
            if n_moves < self.args['exploration_moves']:
                temp = 1.0
            else:
                temp = 0.0

            action = root.select_action(temperature=temp)
            next_state, reward, done = self.game.step(action)
            state = next_state

            n_moves += 1

            if done:
                # Add the terminal state
                examples = []
                for history_state, history_action_probs, history_player in train_examples:
                    examples.append((history_state, history_action_probs,
                                        -reward if history_player == self.game.to_play() else reward))
                return examples


class Trainer:
    def __init__(self, game, model, args, num_workers=4):
        self.game = game
        self.model = model
        self.args = args

        self.num_workers = num_workers
        self.simulations = [Simulation.remote(self.game, self.args) for _ in range(self.num_workers)]

        # get some sampled states and their associated winning move and value.
        self.state_space, self.policy_space, self.value_space, self.masks = get_states_policies_values_masks(game.num_piles,
                                                                                                             num_samples=self.args['num_samples'])

    def learn(self):
        for i in range(1, self.args['numIters'] + 1):
            print(f'{i}', end=" ")
            train_examples = []
            for i in range(self.args['numEps'] // self.num_workers):
                examples = ray.get([sim.execute_episode.remote(self.model) for sim in self.simulations])
                for exp in examples:
                    train_examples.extend(exp)
            self.train(train_examples)

            _, p_acc, v_acc = self.eval_policy_value_acc()
            print(f"{p_acc:.2f}", f"{v_acc:.2f}", len(self.model.policy_head))

    def train(self, examples):
        for board, target_pi, target_v in examples:
            self.model.update(board, target_pi, target_v)


    def eval_policy_value_acc(self, branching_factor=1, value_threshold=1.0):
        p_acc = 0
        random_p_acc = 0
        p_total = 0

        v_acc = 0
        v_total = 0
        for state, policies_target, value_target, mask in zip(self.state_space, self.policy_space, self.value_space, self.masks):
            probs, value = self.model.inference(np.array(state))
            probs = probs * np.array(mask)
            random_probs = np.random.rand(len(probs)) * np.array(mask)
            # if this state has at least one winning move
            if len(policies_target) > 0:
                # calculate the accuracy of the policy obtained from the policy network
                for policy in policies_target:
                    policy = np.array(policy, dtype=np.float32)
                    indicies = probs.argsort()[-branching_factor:].tolist()
                    # if the move corresponding to the highest probability is (one of) the winning move
                    if np.where(policy == 1.0)[0][0] in indicies:
                        p_acc += 1
                        break
                # calculate the accuracy of the random policy
                for policy in policies_target:
                    policy = np.array(policy, dtype=np.float32)
                    random_indices = random_probs.argsort()[-branching_factor:].tolist()
                    if np.where(policy == 1.0)[0][0] in random_indices:
                        random_p_acc += 1
                        break

                p_total += 1
            # calculate the accuracy of the random value
            if abs(value_target - value) < value_threshold:
                v_acc += 1
            v_total += 1

        return float(random_p_acc/p_total), float(p_acc/p_total), float(v_acc/v_total)

            
            
            