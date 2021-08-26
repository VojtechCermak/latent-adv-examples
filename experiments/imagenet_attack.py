import os
import sys
import json
import torch
import pickle
import numpy as np
import argparse
from datetime import datetime

sys.path.insert(0, "../")
from utils import fix_seed
from models import pretrained
import methods

classifiers = {
    'imagenet-effnet': pretrained.EfficientNetClassifier
}

generators = {
    'imagenet-bigbigan' :  pretrained.ImagenetBigBiGAN,
}

attacks = {
    'penalty_pop': methods.PenaltyPopMethod,
    'penalty':     methods.PenaltyMethod,
    'projection':  methods.ProjectionMethod,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest="path_input", help = 'path to experiment JSON', required=True)
    parser.add_argument('-z', dest="path_z", help = 'path to .npy file with encoded z', required=True)
    parser.add_argument('-o', dest="path_output", help = 'path to output folder', default='.')
    parser.add_argument('-device', dest="device", default='cuda')
    parser.add_argument('-seed', dest="seed", type=int, default=1)
    args = parser.parse_args()
    fix_seed(args.seed)

    # Create output folder
    name = args.path_input.split('/')[-1].split('.')[0]
    folder = os.path.join(args.path_output, name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Load data
    z_samples = np.load(args.path_z)
    with open(args.path_input) as json_file:
        experiments = json.load(json_file)
    
    # Run experiments
    for i, experiment in enumerate(experiments):
        generator = generators[experiment['generator']](experiment['generator_level'], args.device)
        classifier = classifiers[experiment['classifier']](args.device)
        method = attacks[experiment['method']](**experiment['params'])
        combined = lambda z: classifier(generator.decode(z))

        results = {}
        for target in experiment['targets']:
            target_labels = torch.tensor([target], dtype=torch.int64, device=args.device)

            perturbed = []
            for sample in z_samples:
                v0 = generator.encode(torch.tensor(sample).to(args.device).unsqueeze(0))
                v_per = method(x0=v0, x_init=None, classifier=combined, generator=generator, target=target_labels)
                perturbed.append(generator.decode(v_per).cpu().detach())
                del v0, v_per
            results[target] = torch.cat(perturbed).numpy()

        data = {
            'script': vars(args),
            'params': method.params_all,
            'json':   experiment,
            'x_per':  results,
        }

        # Save results
        time = datetime.now().strftime('%b%d-%H-%M-%S')
        file =  os.path.join(folder, f"{name}_{i}_{time}.pickle")
        with open(file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'Part {i} done')