import os
import sys
import torch
import pickle
import argparse

sys.path.insert(0, "../")
from utils import class_sampler, sample_grid, fix_seed
from models import pretrained

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', dest="generator", help = 'name of generator', required=True)
    parser.add_argument('-c', dest="classifier", help = 'name of classifier', required=True)
    parser.add_argument('-o', dest="path_output", help = 'path to output folder', default='.')
    parser.add_argument('-m', dest="method",  required=True)
    parser.add_argument('-device', dest="device", default='cuda')
    parser.add_argument('-seed', dest="seed", type=int, default=1)
    args = parser.parse_args()

    generator = generators[args.generator]('full', device=args.device)
    classifier = classifiers[args.classifier](device=args.device)

    if args.method == 'grid': # sample grid of pairs
        fix_seed(args.seed)
        z0, z = sample_grid(classifier, generator, device=args.device)
    elif args.method == 'row': # sample one for each class
        fix_seed(args.seed+1)
        z0 = torch.cat([class_sampler(classifier, generator, c, samples=1, device='cuda') for c in range(10)])
        z = torch.roll(z0, 2, dims=0) 
    else:
        raise ValueError('Invalid method')


    data = {
        'script': vars(args),
        'z0':     z0.cpu().detach().numpy(),
        'z':      z.cpu().detach().numpy(),
        'x0':     generator.decode(z0).cpu().detach().numpy(),
        'x':      generator.decode(z).cpu().detach().numpy(),
        'y0':     classifier(generator.decode(z0)).argmax(1).cpu().detach().numpy(),
        'y':      classifier(generator.decode(z)).argmax(1).cpu().detach().numpy(),
    }
    # Create output folder
    folder = os.path.join(args.path_output)
    if not os.path.exists(folder):
        os.makedirs(folder)

    file =  os.path.join(folder, f"sampled_data_{args.method}.pickle")
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)