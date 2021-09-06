import os
import sys
import torch
import pickle
import argparse
import foolbox as fb
import eagerpy as ep
sys.path.insert(0, "../")
from utils import fix_seed
from models.pretrained import MnistALI, MnistClassifier, MnistMadryRobust
from methods import ProjectionMethod

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="path_input", help = 'path to pickle with input data', required=True) # Row data
    parser.add_argument('-o', dest="path_output", help = 'path to output folder', default='results')
    parser.add_argument('-seed', dest="seed", type=int, default=1)
    args = parser.parse_args()
    fix_seed(args.seed)

    print('Preparing')
    classifier_natural = MnistClassifier().eval()
    classifier_robust = MnistMadryRobust().eval()

    with open(args.path_input, 'rb') as handle:
        data = pickle.load(handle)
    x0 = torch.tensor(data['x0'], device='cuda')
    y0 = torch.tensor(data['y0'], device='cuda')
    z0 = torch.tensor(data['z0'], device='cuda')
    z = torch.tensor(data['z'], device='cuda')

    results = {}
    results['original'] = x0.detach().cpu().numpy()
    
    print('Starting CW attacks')
    attack = fb.attacks.L2CarliniWagnerAttack()
    crit = fb.Misclassification(y0)
    images, = ep.astensors(x0.detach())

    fmodel = fb.PyTorchModel(classifier_natural, bounds=(0, 1))
    _, advs, success = attack(model=fmodel, inputs=images, criterion=crit, epsilons=[100.0])
    assert success.all()
    results['natural_cw_l2'] = advs[-1].raw.detach().cpu().numpy()

    fmodel = fb.PyTorchModel(classifier_robust, bounds=(0, 1))
    _, advs, success = attack(model=fmodel, inputs=images, criterion=crit, epsilons=[100.0])
    assert success.all()
    results['robust_cw_l2'] = advs[-1].raw.detach().cpu().numpy()


    print('Starting latent attacks')
    partial_generator = MnistALI(2)
    combined_natural = lambda z: classifier_natural(partial_generator.decode(z))
    combined_robust = lambda z: classifier_robust(partial_generator.decode(z))
    v0 = partial_generator.encode(z0)
    v = partial_generator.encode(z)

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

    # natural - l2
    method = ProjectionMethod(**params_l2)
    z_per = method(v0, v, combined_natural, partial_generator)
    results['natural_latent_l2'] = partial_generator.decode(z_per).detach().cpu().numpy()

    # natural - wd
    method = ProjectionMethod(**params_wd)
    z_per = method(v0, v, combined_natural, partial_generator)
    results['natural_latent_wd']  = partial_generator.decode(z_per).detach().cpu().numpy()

    # robust - l2
    method = ProjectionMethod(**params_l2)
    z_per = method(v0, v, combined_robust, partial_generator)
    results['robust_latent_l2'] = partial_generator.decode(z_per).detach().cpu().numpy()

    # robust - wd
    method = ProjectionMethod(**params_wd)
    z_per = method(v0, v, combined_robust, partial_generator)
    results['robust_latent_wd']  = partial_generator.decode(z_per).detach().cpu().numpy()

    file =  os.path.join(args.path_output, f"compare_differences.pickle")
    with open(file, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)