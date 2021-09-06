import os
import sys
import torch
import pickle
import argparse
import foolbox as fb
import eagerpy as ep
sys.path.insert(0, "../")
from utils import fix_seed
from models.pretrained import MnistClassifier, MnistMadryRobust

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="path_input", help = 'path to pickle with input data', required=True)  # Grid data
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
    target = torch.tensor(data['y'], device='cuda')


    print('Starting attacks')
    results = {}
    results['original'] = x0.detach().cpu().numpy()
    attack = fb.attacks.L2CarliniWagnerAttack()
    images, = ep.astensors(x0.detach())

    # Untargeted
    crit = fb.Misclassification(y0)
    fmodel = fb.PyTorchModel(classifier_natural, bounds=(0, 1))
    _, advs, success = attack(model=fmodel, inputs=images, criterion=crit, epsilons=[100.0])
    assert success.all()
    results['natural_cw_untargeted'] = advs[-1].raw.detach().cpu().numpy()

    fmodel = fb.PyTorchModel(classifier_robust, bounds=(0, 1))
    _, advs, success = attack(model=fmodel, inputs=images, criterion=crit, epsilons=[100.0])
    assert success.all()
    results['robust_cw_untargeted'] = advs[-1].raw.detach().cpu().numpy()

    # Targeted
    crit = fb.TargetedMisclassification(target)
    fmodel = fb.PyTorchModel(classifier_natural, bounds=(0, 1))
    _, advs, success = attack(model=fmodel, inputs=images, criterion=crit, epsilons=[100.0])
    assert success.all()
    results['natural_cw_targeted'] = advs[-1].raw.detach().cpu().numpy()

    fmodel = fb.PyTorchModel(classifier_robust, bounds=(0, 1))
    _, advs, success = attack(model=fmodel, inputs=images, criterion=crit, epsilons=[100.0])
    assert success.all()
    results['robust_cw_targeted'] = advs[-1].raw.detach().cpu().numpy()

    file =  os.path.join(args.path_output, f"compare_cw_attacks.pickle")
    with open(file, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)