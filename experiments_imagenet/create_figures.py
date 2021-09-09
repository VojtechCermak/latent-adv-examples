import os
import pickle
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from torchvision.utils import make_grid
from pytorch_pretrained_gans import make_gan

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="path_input", help = 'path to folder with results', default='results')
    parser.add_argument('-o', dest="path_output", help = 'path to output folder', default='figures')
    args = parser.parse_args()

    model = make_gan(gan_type='bigbigan')
    model = model.eval().to('cpu')
    z_data = torch.tensor(np.load("z_animals.npy"))
    x_data = ((model(z_data) + 1) / 2)
    z_data = torch.tensor(np.load("z_broccoli.npy"))
    x_broccoli = ((model(z_data) + 1) / 2)

    # 1. Basic plots
    for lvl in [0, 1, 2, 3]:
        path = os.path.join(args.path_input,  f'imagenet_lvl{lvl}.pickle')
        with open(path, 'rb') as handle:
            data = pickle.load(handle)

        for j in range(3):
            x_per = data['x_per'][j]
            diff = (torch.abs(x_per-x_data)**(1/2))
            diff = 1-diff.mean(1, keepdim=True).repeat(1, 3, 1, 1)
            results = torch.cat([x_data, x_per, diff])

            # Plot
            grid = make_grid(results, nrow=5, normalize=False)
            grid = grid.cpu().detach().numpy()
            grid = np.transpose(grid, (1,2,0))
            fig = plt.figure(figsize=(10, 15))
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            ax.imshow(grid)
            fig.savefig(os.path.join(args.path_output, f'imagenet_results_lvl{lvl}_{j}.png'), bbox_inches='tight')


    # 2. Development
    path = os.path.join(args.path_input,  'imagenet_lvl2.pickle')
    with open(path, 'rb') as handle:
        data = pickle.load(handle)

    select = [0, 1, 3, 6, 99]
    animal_id = 0
    for broccoli_id in range(len(x_broccoli)):
        # Figure
        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(1, len(select)+2)
        gs.update(wspace=0.05, hspace=-0.15)

        # Plot
        sequence = data['intermediate'][broccoli_id][animal_id]
        grid = make_grid(x_broccoli[[broccoli_id]], nrow=1, normalize=False)
        grid = grid.cpu().detach().numpy()
        grid = np.transpose(grid, (1,2,0))
        ax = fig.add_subplot(gs[:, 0])
        ax.set_xlabel('Initialization', fontsize=14)
        ax.xaxis.set_label_position('top') 
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(grid)

        for i in range(len(select)):
            grid = make_grid(sequence[select[i]], nrow=4, normalize=False)
            grid = grid.cpu().detach().numpy()
            grid = np.transpose(grid, (1,2,0))
            ax = fig.add_subplot(gs[i+1])
            ax.set_xlabel(f'Iteration {(select[i]*10)}', fontsize=14)
            ax.xaxis.set_label_position('top') 
            ax.set_yticks([])
            ax.set_xticks([])
            ax.imshow(grid)

        grid = make_grid(x_data[[animal_id]], nrow=1, normalize=False)
        grid = grid.cpu().detach().numpy()
        grid = np.transpose(grid, (1,2,0))
        ax = fig.add_subplot(gs[-1])
        ax.set_xlabel('Original image', fontsize=14)
        ax.xaxis.set_label_position('top') 
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(grid)
        fig.savefig(os.path.join(args.path_output, f'imagenet_development_init{broccoli_id}.png'), bbox_inches='tight')


    # 3. Development
    animals = [0, 1, 2, 3, 4] # animal indexes
    inits   = [0, 1, 2]       # broccolis
    levels  = [0, 1, 2, 3]    # generator levels
    for animal in animals:
        results = []
        for j in inits:
            for i in levels:
                path = os.path.join(args.path_input,  f'imagenet_lvl{i}.pickle')
                with open(path, 'rb') as handle:
                    data = pickle.load(handle)
                x_per = data['x_per'][j][animal]
                results.append(x_per.cpu())
            results.append(x_data[animal].cpu())
        results = torch.stack(results)

        # Plot
        grid = make_grid(results, nrow=5, normalize=False)
        grid = grid.cpu().detach().numpy()
        grid = np.transpose(grid, (1, 2, 0))
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.imshow(grid)
        fig.savefig(os.path.join(args.path_output, f'imagenet_levels_animal{animal}.png'), bbox_inches='tight')

    # 4. Levels detail
    levels = [1, 2, 3] # generator levels
    animal = 3
    init = 2
    results = []
    for i in levels:
        path = os.path.join(args.path_input,  f'imagenet_lvl{i}.pickle')
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        x_per = data['x_per'][init][animal]

        diff = (torch.abs(x_per-x_data[[animal]])**(1/2))
        diff = 1-diff.mean(1, keepdim=True).repeat(1, 3, 1, 1)[0]
        results.append(torch.stack([x_per, diff]).cpu())
    results = torch.cat(results)
    order = [i for i in range(len(results)) if i%2==0] + [i for i in range(len(results)) if (i+1)%2==0]
    results = results[order]

    # Plot
    grid = make_grid(results, nrow=len(levels), normalize=False)
    grid = grid.cpu().detach().numpy()
    grid = np.transpose(grid, (1, 2, 0))
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.imshow(grid)
    fig.savefig(os.path.join(args.path_output, f'imagenet_levels_detail.png'), bbox_inches='tight')



    # Sample init vs levels.
    inputs = [
        {'init': 0, 'animal':0},
        {'init': 1, 'animal':0},
        {'init': 0, 'animal':3},
        {'init': 1, 'animal':3},
    ]
    levels = [1, 2, 3]
    results = []
    for d in inputs:
        x_per = []
        for l in levels:
            path = os.path.join(args.path_input,  f"imagenet_lvl{l}.pickle")
            with open(path, 'rb') as handle:
                data = pickle.load(handle)
            x_per.append(data['x_per'][d['init']][d['animal']])
        results.append(torch.stack([x_broccoli[d['init']], *x_per, x_data[d['animal']]]))
    results = torch.cat(results)

    grid = make_grid(results, nrow=len(levels)+2, normalize=False)
    grid = grid.cpu().detach().numpy()
    grid = np.transpose(grid, (1, 2, 0))
    fig = plt.figure(figsize=(15, 15))

    ax = fig.add_subplot(111)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(64, grid.shape[1]-64, len(levels)+2))
    ax.set_xticklabels(['Initialization', *[f'Adversarial (layer {i})' for i in levels], 'Original'])
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('top')
    ax.imshow(grid)
    fig.savefig(os.path.join(args.path_output, f'imagenet_levels_inits.png'), bbox_inches='tight')
