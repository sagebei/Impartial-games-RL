import numpy as np
import random
from NimEnvironments import NimEnv
from model import Nim_Model
from trainer import Trainer
import multiprocessing
import ray


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)
    num_workers = 50  # multiprocessing.cpu_count() - 1

    args = {
        'piles': 5,  # 6, 7
        'num_simulations': 100,  # 70, 100
        'numEps': 104,
        'numIters': 2000,
        'exploration_moves': 3,
        'num_samples': 10000,
        'alpha': 0.35,
        'epsilon': 0.25
    }
    
    train_id = "_".join(str(p) if not isinstance(p, list) else "m".join(str(i) for i in p) for p in args.values())

    game = NimEnv(num_piles=args['piles'])
    model = Nim_Model(action_size=game.action_size, lr=0.001)  # 0.002   73

    trainer = Trainer(game, model, args, num_workers=num_workers)
    trainer.learn()

    writer.close()
    ray.shutdown()
