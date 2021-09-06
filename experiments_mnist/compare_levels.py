import os
import sys
import torch
import pickle
import argparse
sys.path.insert(0, "../")
from utils import fix_seed
from models.pretrained import MnistALI, MnistClassifier
from methods import ProjectionMethod

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="path_input", help = 'path to pickle with input data', required=True)  # Row data
    parser.add_argument('-o', dest="path_output", help = 'path to output folder', default='results')
    parser.add_argument('-seed', dest="seed", type=int, default=1)
    args = parser.parse_args()
    fix_seed(args.seed)

    print('Preparing')
    generator = MnistALI()
    classifier = MnistClassifier().eval()

    with open(args.path_input, 'rb') as handle:
        data = pickle.load(handle)
    z0 = torch.tensor(data['z0'], device='cuda')
    z = torch.tensor(data['z'], device='cuda')

    print('Starting latent attacks')
    params_wd = {
        'iters': 1000,
        'distance': 'wd',
        'distance_args': {'loss': 'sinkhorn', 'p': 1, 'blur': 0.1, 'scaling': 0.5},
        'constraint': 'misclassify',
        'xi_c': {'scheduler': 'SchedulerExponential',
        'params': {'initial': 1, 'gamma': 0.005}},
        'xi_o': {'scheduler': 'SchedulerConstant', 'params': {'alpha': 1}},
        'threshold': 1
        }

    params_l2 = {
        'iters': 1000,
        'distance': 'l2',
        'constraint': 'misclassify',
        'xi_c': {'scheduler': 'SchedulerExponential',
        'params': {'initial': 1, 'gamma': 0.005}},
        'xi_o': {'scheduler': 'SchedulerConstant', 'params': {'alpha': 1}},
        'threshold': 1
        }

    results = {}
    results['script'] = vars(args)
    results['original'] = data['x0']

    levels = ['full', 0, 1, 2, 3]
    for level in levels:
        print(f"Starting level: {level}")
        gp_generator = MnistALI(level)
        gp_combined = lambda z: classifier(gp_generator.decode(z))
        v0 = gp_generator.encode(z0)
        v = gp_generator.encode(z)

        method = ProjectionMethod(**params_l2)
        z_per = method(v0, v, gp_combined, gp_generator)
        results[f"l2_level_{level}"] = gp_generator.decode(z_per).detach().cpu()

        method = ProjectionMethod(**params_wd)
        z_per = method(v0, v, gp_combined, gp_generator)
        results[f"wd_level_{level}"] = gp_generator.decode(z_per).detach().cpu()

    file =  os.path.join(args.path_output, f"compare_levels.pickle")
    with open(file, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)