import os
import sys
import json
import torch
import pickle
import argparse
from datetime import datetime

sys.path.insert(0, "../")
from utils import fix_seed
from models import pretrained
import methods


classifiers = {
    'mnist-cls':     pretrained.MnistClassifier,
    'madry-robust':  pretrained.MnistMadryRobust,
    'madry-natural': pretrained.MnistMadryNatural,
    'svhn-cls':      pretrained.SvhnClassifier,
    'cifar-cls':     pretrained.CifarClassifier,
}

generators = {
    'mnist-ali' :  pretrained.MnistALI,
    'mnist-dcgan': pretrained.MnistDCGAN,
    'mnist-vae':   pretrained.MnistVAE,
    'svhn-ali':    pretrained.SvhnALI,
    'svhn-dcgan':  pretrained.SvhnDCGAN,
    'svhn-vae':    pretrained.SvhnVAE,
    'cifar-ali':   pretrained.CifarALI,
    'cifar-dcgan': pretrained.CifarDCGAN,
}

attacks = {
    'penalty_pop': methods.PenaltyPopMethod,
    'penalty':     methods.PenaltyMethod,
    'projection':  methods.ProjectionMethod,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest="path_input", help = 'path to experiment JSON', required=True)
    parser.add_argument('-i', dest="path_pickle", help = 'path to pickle with input data', required=True)  # Grid data
    parser.add_argument('-o', dest="path_output", help = 'path to output folder', default='results')
    parser.add_argument('-device', dest="device", default='cuda')
    parser.add_argument('-seed', dest="seed", type=int, default=1)
    args = parser.parse_args()


    # Create output folder
    folder = os.path.join(args.path_output)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(args.path_pickle, 'rb') as handle:
        data = pickle.load(handle)
    z0 = torch.tensor(data['z0'], device=args.device)
    z = torch.tensor(data['z'], device=args.device)

    with open(args.path_input) as json_file:
        experiments = json.load(json_file)

    print('Starting experiments')
    for i, experiment in enumerate(experiments):
        fix_seed(args.seed)
        start = datetime.now()
        construct_generator = generators[experiment['generator']]
        method = attacks[experiment['method']](**experiment['params'])
        classifier = classifiers[experiment['classifier']]()

        # Construct latent vectors z
        generator = construct_generator(experiment['generator_level'])
        combined = lambda z: classifier(generator.decode(z))

        # Construct latent vectors v
        v0 = generator.encode(z0)
        v = generator.encode(z)
        target = combined(v).argmax(1)
        v_per = method(x0=v0, x_init=v, classifier=combined, generator=generator, target=target)

        # Collect results
        x0 = generator.decode(v0)
        x = generator.decode(v)
        x_per = torch.zeros_like(x)
        have_nan = torch.isnan(v_per).view(v_per.shape[0], -1).any(dim=1)
        x_per[~have_nan] = generator.decode(v_per[~have_nan])

        data = {
            'time':   str(datetime.now() - start),
            'script': vars(args),
            'params': method.params_all,
            'json':   experiment,
            'z0':     z0.cpu().detach().numpy(),
            'z':      z.cpu().detach().numpy(),
            'x0':     x0.cpu().detach().numpy(),
            'x':      x.cpu().detach().numpy(),
            'x_per':  x_per.cpu().detach().numpy(),
            'v_per':  v_per.cpu().detach().numpy(),
        }

        # Save results
        file = os.path.join(folder, f"{experiment['name']}.pickle")
        with open(file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'Part {i} done')