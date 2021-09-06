import os
import sys
import json
import torch
import pickle
import numpy as np
import argparse

sys.path.insert(0, "../")
from utils import fix_seed
from models import pretrained
from methods import calculate_gradients, ProjectionMethod
from projections import ConvergenceError, ProjectionBinarySearch


class ProjectionMethodContainer(ProjectionMethod):
    '''
    Adds container to store intermediate results of the projection method.
    '''
    def method(self, objective, constraint, x_init, xi_c, xi_o, grad_norm_o='l2', grad_norm_c='l2', iters=50, threshold=1e-3):
        x = x_init.clone()
        projection = ProjectionBinarySearch(constraint, threshold=threshold)

        # Project in the direction of objective
        grad_objective = calculate_gradients(objective, x, norm=grad_norm_o)
        x_next = x - xi_o(0)*grad_objective
        x = projection(x, x_next)

        self.container = []
        for t in range(iters):
            # Step in direction of constraint
            if not (x_next == x).all().item():
                grad_constraint = calculate_gradients(constraint, x, norm=grad_norm_c)
                lr = xi_c(t)
                converged = False
                for _ in range(100):
                    if constraint(x - lr*grad_constraint) < 0 :
                        converged = True
                        break
                    else:
                        lr = lr / 2
                if not converged:
                    raise ConvergenceError("Step correction is in infinite cycle")
                x = x - lr*grad_constraint

            # Project in the direction of objective
            grad_objective = calculate_gradients(objective, x, norm=grad_norm_o)
            x_next = x - xi_o(0)*grad_objective
            x = projection(x, x_next)
            if t % 10 == 0:
                self.container.append(x.detach().cpu())
        return x.detach()

classifiers = {
    'imagenet-effnet': pretrained.EfficientNetClassifier
}

generators = {
    'imagenet-bigbigan' :  pretrained.ImagenetBigBiGAN,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', dest="path_z", help = 'path to .npy file with encoded x0 images', required=True)
    parser.add_argument('-i', dest="path_i", help = 'path to .npy file with encoded x_init images', required=True)
    parser.add_argument('-p', dest="path_input", help = 'path to experiment JSON', required=True)
    parser.add_argument('-o', dest="path_output", help = 'path to output folder', default='.')
    parser.add_argument('-device', dest="device", default='cuda')
    parser.add_argument('-seed', dest="seed", type=int, default=1)
    args = parser.parse_args()
    fix_seed(args.seed)

    # Create output folder
    folder = os.path.join(args.path_output)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Load data
    z_init = np.load(args.path_i)
    z_samples = np.load(args.path_z)
    with open(args.path_input) as json_file:
        experiments = json.load(json_file)

    # Run experiments
    for j, experiment in enumerate(experiments):
        generator = generators[experiment['generator']](experiment['generator_level'], args.device)
        classifier = classifiers[experiment['classifier']](args.device)
        combined = lambda z: classifier(generator.decode(z))
        target_labels = torch.tensor([experiment['target']], dtype=torch.int64, device=args.device)

        results = {}
        containers = {}
        for i, init in enumerate(z_init):
            perturbed = []
            container = []
            for sample in z_samples:
                method = ProjectionMethodContainer(**experiment['params'])
                v0 = generator.encode(torch.tensor(sample).to(args.device).unsqueeze(0))
                v_init = generator.encode(torch.tensor(init).to(args.device).unsqueeze(0))
                v_per = method(x0=v0, x_init=v_init, classifier=combined, generator=generator, target=target_labels)

                collected = [generator.decode(torch.tensor(v, device=args.device)).cpu().detach() for v in method.container]
                container.append(torch.cat(collected))
                perturbed.append(generator.decode(v_per).cpu().detach())
                del v0, v_per, v_init
            results[i] = torch.cat(perturbed)
            containers[i] = container
            print(f'Part {j}-{i} done')

        data = {
            'script': vars(args),
            'params': method.params_all,
            'json':   experiment,
            'x_per':  results,
            'intermediate': containers,
        }

        # Save results
        file = os.path.join(folder, f"{experiment['name']}.pickle")
        with open(file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
