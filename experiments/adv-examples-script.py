import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geomloss import SamplesLoss
import argparse

# Custom library
sys.path.insert(0, "../")
from utils import class_sampler, grid_plot, batch_add_lsb
from models.pretrained import MnistALI, MnistClassifier
from schedulers import *
from constraints import *
from objectives import *
from distances import *
from methods import *
from projections import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', dest="path_output", help = 'path to output folder', default='.')
    parser.add_argument('-device', dest="device", default='cuda')
    parser.add_argument('-seed', dest="seed", type=int, default=1)
    parser.add_argument('-lsb', dest="add_lsb", type=int, default=2)
    args = parser.parse_args()

    # Reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load models
    device = args.device
    image_folder = args.path_output
    add_lsb = args.add_lsb
    generator = MnistALI()
    classifier = MnistClassifier()
    combined = lambda z: classifier(generator.decode(z))

    # Sample data
    torch.manual_seed(1)
    z_data = torch.cat([class_sampler(classifier, generator, c, samples=2, threshold=0.99, device=device) for c in range(10)])
    y_data = torch.tensor([[c]*2 for c in range(10)], device=device).flatten()
    x_data = generator.decode(z_data)
    z0 = z_data.clone()
    z = torch.roll(z0.clone(), 2, dims=0)


    # No attack
    grid_plot(batch_add_lsb(x_data, add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/no_attack.png')

    ## 1. Perturbations in pixels
    # 1. Image - FGSM Attack
    x_per = fgsm(x_data, y_data, classifier, epsilon=0.3)
    grid_plot(batch_add_lsb(x_per, add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/images_fgsm.png')

    # 1. Images - Untargeted PGD Attack
    objective = Objective(y_data, nn.CrossEntropyLoss(), classifier, targeted=False)
    projection = ProjectionLinf(0.3)
    x_per = projected_gd(x_data, objective, projection, grad_norm='sign', steps=50, step_size=0.1, clip=(0, 1))
    grid_plot(batch_add_lsb(x_per, add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/images_pgd_untargeted.png')

    # 1. Images - Targeted PGD Attack
    objective = Objective(y_data*0, nn.CrossEntropyLoss(), classifier, targeted=True)
    projection = ProjectionLinf(0.3)
    x_per = projected_gd(x_data, objective, projection, grad_norm='sign', steps=50, step_size=0.1, clip=(0, 1))
    grid_plot(batch_add_lsb(x_per, add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/images_pgd_targeted.png')

    ## 2. Perturbations in latent space
    # 2. Latent - minimize losss (CE) such that ||x0-x||2 < epsilon
    distance = L2(Decoded(generator))
    constraint = ConstraintDistance(z_data, distance, 5.0)
    projection = ProjectionBinarySearch(constraint, threshold=0.001)
    objective = Objective(y_data, nn.CrossEntropyLoss(), combined, targeted=False)
    z_per = projected_gd(z_data, objective, projection, grad_norm='sign', steps=20, step_size=0.1, clip=None)
    grid_plot(batch_add_lsb(generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/latent_loss_l2.png')

    # 2. Latent - minimize losss (CE) such that WD(x0, x1) < epsilon
    distance = GeomLoss(SamplesLoss("sinkhorn", p=1, blur=0.1, scaling=0.5), DecodedDistribution(generator))
    constraint = ConstraintDistance(z_data, distance, 0.0025)
    projection = ProjectionBinarySearch(constraint, threshold=0.0005)
    objective = Objective(y_data, nn.CrossEntropyLoss(), combined, targeted=False)
    z_per = projected_gd(z_data, objective, projection, grad_norm='l2', steps=20, step_size=0.1, clip=None)
    grid_plot(batch_add_lsb(generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/latent_loss_wd.png')

    # 2. Latent - simple projection
    z_per = bisection_method(z0, z, combined)
    grid_plot(batch_add_lsb(generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/latent_simple_projection.png')

    # 2. Latent - penalty method l2
    rho = SchedulerStep(10e8, gamma=1, n=10)
    xi = SchedulerExponential(initial=1, gamma=0.01)
    distance = L2(Decoded(generator))
    z_per = penalty_method(z0, distance, combined, xi, rho, grad_norm='l2', iters=1000)
    nan = z_per.isnan().all(1).flatten()
    grid_plot(batch_add_lsb(generator.decode(z_per[~nan]), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/latent_penalty_l2.png')
    
    # 2. Latent - penalty method WD
    distance = GeomLoss(SamplesLoss("sinkhorn", p=1, blur=0.1, scaling=0.5), DecodedDistribution(generator))
    rho = SchedulerStep(10e8, gamma=1, n=10)
    xi = SchedulerExponential(initial=1, gamma=0.01)
    z_per = penalty_method(z0, distance, combined, xi, rho, iters=1000)
    grid_plot(batch_add_lsb(generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/latent_penalty_wd.png')

    # 2. Latent - projection method WD
    distance = L2(Decoded(generator))
    xi_o = SchedulerConstant(alpha=1)
    xi_c = SchedulerPower(initial=1, power=-1/2)
    z_per = projection_method(z0, z, distance, combined, xi_c, xi_o, grad_norm_o='l2', grad_norm_c='l2', iters=150)
    grid_plot(batch_add_lsb(generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/latent_projection_l2.png')

    # 2. Latent - projection method WD
    loss_function = SamplesLoss("sinkhorn", p=1, blur=0.1, scaling=0.5)
    distance = GeomLoss(loss_function, DecodedDistribution(generator))
    xi_o = SchedulerConstant(alpha=1)
    xi_c = SchedulerPower(initial=1, power=-1/2)
    z_per = projection_method(z0, z, distance, combined, xi_c, xi_o, grad_norm_o='l2', grad_norm_c='l2', iters=150)
    grid_plot(batch_add_lsb(generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/latent_projection_wd.png')


    # 3. Partial generators
    # 3. Level 0
    xi_o = SchedulerConstant(alpha=1)
    xi_c = SchedulerPower(initial=1, power=-1/2)
    gp_generator = MnistALI(0)
    gp_combined = lambda z: classifier(gp_generator.decode(z))
    v0 = gp_generator.encode(z0)
    v = gp_generator.encode(z)

    distance = L2(Decoded(gp_generator))
    z_per = projection_method(v0, v, distance, gp_combined, xi_c, xi_o, iters=150)
    grid_plot(batch_add_lsb(gp_generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/partial0_projection_l2.png')

    loss_function = SamplesLoss("sinkhorn", p=1, blur=0.1, scaling=0.5)
    distance = GeomLoss(loss_function, DecodedDistribution(gp_generator))
    z_per = projection_method(v0, v, distance, gp_combined, xi_c, xi_o, iters=150)
    grid_plot(batch_add_lsb(gp_generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/partial0_projection_wd.png')

    # 3. Level 1
    xi_o = SchedulerConstant(alpha=1)
    xi_c = SchedulerPower(initial=1, power=-1/2)
    gp_generator = MnistALI(1)
    gp_combined = lambda z: classifier(gp_generator.decode(z))
    v0 = gp_generator.encode(z0)
    v = gp_generator.encode(z)

    distance = L2(Decoded(gp_generator))
    z_per = projection_method(v0, v, distance, gp_combined, xi_c, xi_o, iters=150)
    grid_plot(batch_add_lsb(gp_generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/partial1_projection_l2.png')

    loss_function = SamplesLoss("sinkhorn", p=1, blur=0.1, scaling=0.5)
    distance = GeomLoss(loss_function, DecodedDistribution(gp_generator))
    z_per = projection_method(v0, v, distance, gp_combined, xi_c, xi_o, iters=150)
    grid_plot(batch_add_lsb(gp_generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/partial1_projection_wd.png')


    # 3. Level 2
    xi_o = SchedulerConstant(alpha=1)
    xi_c = SchedulerPower(initial=1, power=-1/2)
    gp_generator = MnistALI(2)
    gp_combined = lambda z: classifier(gp_generator.decode(z))
    v0 = gp_generator.encode(z0)
    v = gp_generator.encode(z)

    distance = L2(Decoded(gp_generator))
    z_per = projection_method(v0, v, distance, gp_combined, xi_c, xi_o, iters=150)
    grid_plot(batch_add_lsb(gp_generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/partial2_projection_l2.png')

    loss_function = SamplesLoss("sinkhorn", p=1, blur=0.1, scaling=0.5)
    distance = GeomLoss(loss_function, DecodedDistribution(gp_generator))
    z_per = projection_method(v0, v, distance, gp_combined, xi_c, xi_o, iters=150)
    grid_plot(batch_add_lsb(gp_generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/partial2_projection_wd.png')


    # 3. Level 3
    xi_o = SchedulerConstant(alpha=1)
    xi_c = SchedulerPower(initial=1, power=-1/2)
    gp_generator = MnistALI(3)
    gp_combined = lambda z: classifier(gp_generator.decode(z))
    v0 = gp_generator.encode(z0)
    v = gp_generator.encode(z)

    distance = L2(Decoded(gp_generator))
    z_per = projection_method(v0, v, distance, gp_combined, xi_c, xi_o, iters=150)
    grid_plot(batch_add_lsb(gp_generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/partial3_projection_l2.png')

    loss_function = SamplesLoss("sinkhorn", p=1, blur=0.1, scaling=0.5)
    distance = GeomLoss(loss_function, DecodedDistribution(gp_generator))
    z_per = projection_method(v0, v, distance, gp_combined, xi_c, xi_o, iters=150)
    grid_plot(batch_add_lsb(gp_generator.decode(z_per), add_lsb=add_lsb), nrows=10, save_as=f'{image_folder}/partial3_projection_wd.png')
