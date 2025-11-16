import numpy as np
from random import shuffle
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
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
        with torch.no_grad():
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
    def __init__(self, game, model, args, device, num_workers=4):
        self.game = game
        self.model = model
        self.args = args
        self.device = device
        self.batch_counter = 0  # record the number of batch data used during the training process
        self.epoch_counter = 0  # record the number of training epochs

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args['lr'],
                                    weight_decay=args['weight_decay'])
        self.scheduler = MultiStepLR(self.optimizer,
                                     milestones=args['milestones'],
                                     gamma=args['scheduler_gamma'])

        self.num_workers = num_workers
        self.simulations = [Simulation.remote(self.game, self.args) for _ in range(self.num_workers)]
        
        self.state_space, self.policy_space, self.value_space, self.masks = get_states_policies_values_masks(game.num_piles,
                                                                                                             num_samples=self.args['num_samples'])

    def learn(self):
        for i in range(1, self.args['numIters'] + 1):
            print(f'{i}/{self.args["numIters"]}')
            train_examples = []
            self.model.to(torch.device('cpu'))
            self.model.eval()
            for i in range(self.args['numEps'] // self.num_workers):
                examples = ray.get([sim.execute_episode.remote(self.model) for sim in self.simulations])
                for exp in examples:
                    train_examples.extend(exp)
            shuffle(train_examples)
            self.train(train_examples)
            random_policy_acc, policy_acc, value_acc = self.eval_policy_value_acc(branching_factor=1)
            print(policy_acc, value_acc)

    def save_best_model(self):
        self.player_pool.save_best_player()

    def train(self, examples):
        for _ in range(self.args['epochs']):
            batch_idx = 0
            while batch_idx < len(examples) // self.args['batch_size']:
                self.model.to(self.device)
                self.model.train()

                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                boards = torch.FloatTensor(np.array(boards)).contiguous().to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).contiguous().to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).contiguous().to(self.device)

                out_pi, out_v = self.model(boards)
                
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                
                
                self.model.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                self.batch_counter += 1
                batch_idx += 1
                
            self.scheduler.step()
            
                
            self.epoch_counter += 1

    def eval_policy_value_acc(self, branching_factor=1, value_threshold=1.0):
        self.model.eval()
        with torch.no_grad():
            p_acc = 0
            random_p_acc = 0
            p_total = 0

            v_acc = 0
            v_total = 0
            for state, policies_target, value_target, mask in zip(self.state_space, self.policy_space, self.value_space, self.masks):
                probs, value = self.model.predict(np.array(state))
                probs = probs * np.array(mask)
                # if this state has at least one winning move
                hit = False
                if len(policies_target) > 0:
                    # calculate the accuracy of the policy obtained from the policy network
                    for policy in policies_target:
                        policy = np.array(policy, dtype=np.float32)
                        indicies = probs.argsort()[-branching_factor:].tolist()
                        # if the move corresponding to the highest probability is (one of) the winning move
                        for indice in indicies:
                            # print(self.game.all_actions[indice][1])
                            if self.game.all_actions[indice][1] == 1:
                                p_acc += 1
                                hit = True
                                break
                        if hit:
                            break
                    p_total += 1
                    # print(p_acc, p_total)
                # calculate the accuracy of the random value
                if abs(value_target - value) < value_threshold:
                    v_acc += 1
                v_total += 1

            return float(random_p_acc/p_total), float(p_acc/p_total), float(v_acc/v_total)

    # calculate the loss for the policy network
    def loss_pi(self, targets, outputs):
        loss = - (targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    # calculate the loss for the value network
    def loss_v(self, targets, outputs):
        loss = torch.mean((targets - outputs.squeeze()) ** 2)
        return loss
            
            