import os
import pickle
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from torchvision.utils import make_grid
import sys
sys.path.insert(0, "..")

def invert(data):
    return 1-data

def least_significant_bits(sample):
    sample_int = (sample*255).round().type(torch.uint8)
    lsb = torch.remainder(sample_int, 2).type(torch.float)
    return lsb

def plot_original(ax, x_data):
    grid1 = make_grid(invert(x_data), nrow=1, normalize=False)
    grid2 = make_grid(invert(least_significant_bits(x_data)), nrow=1, normalize=False)
    grid = torch.cat([grid1, grid2], 2)
    grid = grid.cpu().detach().numpy()
    grid = np.transpose(grid, (1,2,0))

    ax.set_yticks(np.linspace(16, grid.shape[0]-16, 10))
    ax.set_yticklabels(np.arange(10))
    ax.set_ylabel('Original class')
    ax.set_xticks(np.linspace(16, grid.shape[1]-16, 2))
    ax.set_xticklabels(['Original', 'LSB'])
    ax.set_xlabel('Original images')
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('top')
    ax.imshow(grid)

def plot_block(ax, x_data, x_perturbed, title):
    grid1 = make_grid(invert(x_perturbed), nrow=1, normalize=False)
    grid2 = make_grid(invert(least_significant_bits(x_perturbed)), nrow=1, normalize=False)
    grid3 = make_grid(invert((x_data - x_perturbed).abs()**(1/2)), nrow=1, normalize=False)
    grid = torch.cat([grid1, grid2, grid3], 2)
    grid = grid.cpu().detach().numpy()
    grid = np.transpose(grid, (1,2,0))

    ax.set_yticks([])
    ax.set_xticks(np.linspace(16, grid.shape[1]-16, 3))
    ax.set_xticklabels(['Adversarial', 'LSB', 'Difference'])
    ax.set_xlabel(title)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('top')
    ax.imshow(grid)

def plot_matrix(ax, path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    images = torch.tensor(data['x_per'])
    frame = torch.zeros(100, 1, 28, 28)
    frame[list(set(np.arange(100)) - set(np.arange(0, 100, 11)))] = images

    grid = make_grid(invert(frame), nrow=10, normalize=False)
    grid = grid.cpu().detach().numpy()
    grid = np.transpose(grid, (1,2,0))

    ax.set_yticks(np.linspace(16, grid.shape[0]-16, 10))
    ax.set_yticklabels(np.arange(10))
    ax.set_ylabel('Original class')
    ax.set_xticks(np.linspace(16, grid.shape[0]-16, 10))
    ax.set_xticklabels(np.arange(10))
    ax.set_xlabel('Target class')
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('top')
    ax.imshow(grid)

def plot_untargeted_matrix(path, fig):
    from models.pretrained import MnistClassifier
    classifier = MnistClassifier()
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    x_per = torch.tensor(data['x_per'], device='cuda')
    labels = classifier(x_per).argmax(1)

    gs = gridspec.GridSpec(10, 9)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(x_per):
        ax = fig.add_subplot(gs[i])
        ax.set_xlabel(f'{labels[i].item()}', fontsize=12)
        ax.xaxis.set_label_coords(0.1, 0.95) 
        ax.set_yticks([])
        ax.set_xticks([])
        img = make_grid(img)
        img = 1 - img.cpu().detach()
        ax.imshow(img.permute(1, 2, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="path_input", help = 'path to folder with results', default='results')
    parser.add_argument('-o', dest="path_output", help = 'path to output folder', default='figures')
    args = parser.parse_args()


    # 1. Compare attack structure
    path = os.path.join(args.path_input, 'compare_differences.pickle')
    with open(path, 'rb') as handle:
        data = pickle.load(handle)


    # 1.1 Natural
    x_data = torch.tensor(data['original'])
    x_adv_cw = torch.tensor(data['natural_cw_l2'])
    x_adv_l2 = torch.tensor(data['natural_latent_l2'])
    x_adv_wd = torch.tensor(data['natural_latent_wd'])
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(1, 4)
    gs.update(wspace=0.0)
    ax = fig.add_subplot(gs[0])
    plot_original(ax, x_data)
    ax = fig.add_subplot(gs[1])
    plot_block(ax, x_data, x_adv_cw, 'CW attack')
    ax = fig.add_subplot(gs[2])
    plot_block(ax, x_data, x_adv_l2, 'l2 attack')
    ax = fig.add_subplot(gs[3])
    plot_block(ax, x_data, x_adv_wd, 'Wasserstein attack')
    fig.savefig(os.path.join(args.path_output, 'compare_attacks_natural.png'), bbox_inches='tight')


    # 1.2 Robust
    x_data = torch.tensor(data['original'])
    x_adv_cw = torch.tensor(data['robust_cw_l2'])
    x_adv_l2 = torch.tensor(data['robust_latent_l2'])
    x_adv_wd = torch.tensor(data['robust_latent_wd'])
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(1, 4)
    gs.update(wspace=0.0)
    ax = fig.add_subplot(gs[0])
    plot_original(ax, x_data)
    ax = fig.add_subplot(gs[1])
    plot_block(ax, x_data, x_adv_cw, 'CW attack')
    ax = fig.add_subplot(gs[2])
    plot_block(ax, x_data, x_adv_l2, 'l2 attack')
    ax = fig.add_subplot(gs[3])
    plot_block(ax, x_data, x_adv_wd, 'Wasserstein attack')
    fig.savefig(os.path.join(args.path_output, 'compare_attacks_robust.png'), bbox_inches='tight')


    #3. Targeted matrix
    fig = plt.figure(constrained_layout=True, figsize=(15, 8))
    subfigs = fig.subfigures(1, 2, wspace=0.05)
    ax = subfigs[0].add_subplot()
    path = os.path.join(args.path_input, 'natural-l2-targeted.pickle')
    plot_matrix(ax, path)
    ax = subfigs[1].add_subplot()
    path = os.path.join(args.path_input, 'natural-wd-targeted.pickle')
    plot_matrix(ax, path)
    fig.savefig(os.path.join(args.path_output, 'targeted_matrix.png'), bbox_inches='tight')
            
    fig = plt.figure(figsize=(9, 10))    
    path = os.path.join(args.path_input, 'natural-wd-untargeted.pickle')
    plot_untargeted_matrix(path, fig)
    fig.savefig(os.path.join(args.path_output, 'untargeted_matrix_wd.png'), bbox_inches='tight')

    fig = plt.figure(figsize=(9, 10))    
    path = os.path.join(args.path_input, 'natural-l2-untargeted.pickle')
    plot_untargeted_matrix(path, fig)
    
    fig.savefig(os.path.join(args.path_output, 'untargeted_matrix_l2.png'), bbox_inches='tight')