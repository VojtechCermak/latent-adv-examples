import sys
import json
import torch
import pickle
import argparse
from datetime import datetime

sys.path.insert(0, "../")
from utils import sample_grid
from models import pretrained
import methods


classifiers = {
    'mnist-cls':     pretrained.MnistClassifier,
    'madry-robust':  pretrained.MnistMadryRobust,
    'madry-natural': pretrained.MnistMadryNatural,
    'svhn-cls':      pretrained.SvhnClassifier,
    'cifar-cls':     pretrained.CifarClassifier,
    #TODO:'imagenet-efficientnet-128'
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
    #TODO:'imagenet-bigbigan-128'
}

attacks = {
    'penalty':    methods.penalty_method_wrapped,
    'projection': methods.projection_method_wrapped,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest="path_input", help = 'path to experiment JSON', required=True)
    parser.add_argument('-o', dest="path_output", help = 'path to output folder', default='.')
    parser.add_argument('-classes', dest="no_classes", help = 'number of samples per class', type=int, default=10)
    parser.add_argument('-device', dest="device", default='cuda')
    parser.add_argument('-seed', dest="seed", type=int, default=1)
    parser.add_argument('-sampler_batch_size', dest="sampler_batch_size", type=int, default=32)
    parser.add_argument('-sampler_max_steps', dest="sampler_max_steps", type=int, default=200)
    parser.add_argument('-sampler_threshold', dest="sampler_threshold",  type=float, default=0.99)
    args = parser.parse_args()

    # Reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    params_sampler = {
        'threshold':  args.sampler_threshold,
        'max_steps':  args.sampler_max_steps,
        'batch_size': args.sampler_batch_size,
        'device':     args.device,
    }

    with open(args.path_input) as json_file:
        experiments = json.load(json_file)

    file_data = []
    for i, experiment in enumerate(experiments):
        construct_generator = generators[experiment['generator']]
        classifier = classifiers[experiment['classifier']]()
        method = attacks[experiment['method']]
        
        # Construct latent vectors z
        z0, z, labels = sample_grid(classifier, construct_generator('full'), device='cuda', no_classes=10)
        generator = construct_generator(experiment['generator_level'])
        combined = lambda z: classifier(generator.decode(z))

        # Construct latent vectors v
        v0 = generator.encode(z0)
        v = generator.encode(z)
        v_per = method(v0, v, combined, generator)

        data = {
            'meta':  experiment,
            'args':  vars(args),
            'z0':    z0.cpu().numpy(),
            'z':     z.cpu().numpy(),
            'x0':    generator.decode(v0).cpu().detach().numpy(),
            'x':     generator.decode(v).cpu().detach().numpy(),
            'x_per': generator.decode(v_per).cpu().detach().numpy(),
        }
        file_data.append(data)
        print(f'Part {i} done')

    time = datetime.now().strftime('%b%d-%H-%M-%S')
    name = args.path_input.split('\\')[-1].split('.')[0]
    file_name = args.path_output + '/' + f'{name}_{time}' + '.pickle'

    with open(file_name, 'wb') as handle:
        pickle.dump(file_data, handle, protocol=pickle.HIGHEST_PROTOCOL)  