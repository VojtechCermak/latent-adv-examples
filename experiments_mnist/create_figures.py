import os
import sys
import pickle
import torch
import numpy as np
# Custom library
sys.path.insert(0, "../")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from torchvision.utils import make_grid

def least_significant_bits(sample):
    sample_int = (sample*255).type(torch.uint8)
    lsb = torch.remainder(sample_int, 2).type(torch.float)
    return lsb

def plot_original(ax, x_data):
    grid1 = make_grid(x_data, nrow=1, normalize=False)
    grid2 = make_grid(least_significant_bits(x_data), nrow=1, normalize=False)
    grid = torch.cat([grid1, grid2], 2)
    grid = grid.cpu().detach().numpy()
    grid = np.transpose(grid, (1,2,0))

    ax.set_yticks(np.linspace(16, grid.shape[0]-16, 10))
    ax.set_yticklabels(np.arange(10))
    ax.set_ylabel('Original class')
    ax.set_xticks(np.linspace(16, grid.shape[1]-16, 2))
    ax.set_xticklabels(['Natural', 'LSB'])
    ax.set_xlabel('Encoded images')
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('top')
    ax.imshow(grid)

def plot_block(ax, x_data, x_perturbed, title):
    grid1 = make_grid(x_perturbed, nrow=1, normalize=False)
    grid2 = make_grid(least_significant_bits(x_perturbed), nrow=1, normalize=False)
    grid3 = make_grid((x_data - x_perturbed).abs()**(1/2), nrow=1, normalize=False)
    grid = torch.cat([grid1, grid2, grid3], 2)
    grid = grid.cpu().detach().numpy()
    grid = np.transpose(grid, (1,2,0))

    ax.set_yticks([])
    ax.set_xticks(np.linspace(16, grid.shape[1]-16, 3))
    ax.set_xticklabels(['Perturbed', 'LSB', 'Difference'])
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
    grid = make_grid(frame, nrow=10, normalize=False)
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


# 1. Compare attack structure
path = os.path.join('..', 'experiments', 'results', 'compare_differences.pickle')
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
plot_block(ax, x_data, x_adv_cw, 'CW-l2 attack')
ax = fig.add_subplot(gs[2])
plot_block(ax, x_data, x_adv_l2, 'l-2 perturbation')
ax = fig.add_subplot(gs[3])
plot_block(ax, x_data, x_adv_wd, 'wd perturbation')
fig.savefig('figures/compare_attacks_natural.png', bbox_inches='tight')

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
plot_block(ax, x_data, x_adv_cw, 'CW-l2 attack')
ax = fig.add_subplot(gs[2])
plot_block(ax, x_data, x_adv_l2, 'l-2 perturbation')
ax = fig.add_subplot(gs[3])
plot_block(ax, x_data, x_adv_wd, 'wd perturbation')
fig.savefig('figures/compare_attacks_robust.png', bbox_inches='tight')

# 2. Compare levels
path = os.path.join('..', 'experiments', 'results', 'compare_levels.pickle')
with open(path, 'rb') as handle:
    data = pickle.load(handle)
x_data = torch.tensor(data['original'])
results_wd = [data['wd_level_full'], data['wd_level_0'], data['wd_level_1'], data['wd_level_2'], data['wd_level_3']]
results_l2 = [data['l2_level_full'], data['l2_level_0'], data['l2_level_1'], data['l2_level_2'], data['l2_level_3']]

# Figure
fig = plt.figure(figsize=(17, 10))
gs = gridspec.GridSpec(1, 12)
gs.update(wspace=0.0)

# Original images
grid = make_grid(x_data, nrow=1, normalize=False)
grid = grid.cpu().detach().numpy()
grid = np.transpose(grid, (1,2,0))
ax = fig.add_subplot(gs[:7])
ax.set_yticks(np.linspace(16, grid.shape[0]-16, 10))
ax.set_yticklabels(np.arange(10))
ax.set_xticks(np.linspace(16, grid.shape[0]-16, 1))
ax.set_xticklabels(['Natural'])
ax.set_ylabel('Original class')
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticks_position('top')
ax.imshow(grid)

# Levels - wd
grid = torch.cat([make_grid(value, nrow=1, normalize=False) for value in results_wd], 2)
grid = grid.cpu().detach().numpy()
grid = np.transpose(grid, (1,2,0))
ax = fig.add_subplot(gs[4:8])
ax.set_xticks(np.linspace(16, grid.shape[1]-16, 5))
ax.set_xticklabels(['Latent', 'level 1', 'level 2', 'level 3', 'level 4'])
ax.set_yticks([])
ax.set_xlabel('WD')
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticks_position('top')
ax.imshow(grid)

# Levels - l2
grid = torch.cat([make_grid(value, nrow=1, normalize=False) for value in results_l2], 2)
grid = grid.cpu().detach().numpy()
grid = np.transpose(grid, (1,2,0))
ax = fig.add_subplot(gs[8:12])
ax.set_xticks(np.linspace(16, grid.shape[1]-16, 5))
ax.set_xticklabels(['Latent', 'level 1', 'level 2', 'level 3', 'level 4'])
ax.set_yticks([])
ax.set_xlabel('l2')
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticks_position('top')
ax.imshow(grid)

fig.savefig('figures\compare_levels.png', bbox_inches='tight')

#3. Targeted matrix
fig = plt.figure(constrained_layout=True, figsize=(15, 8))
subfigs = fig.subfigures(1, 2, wspace=0.05)
ax = subfigs[0].add_subplot()
subfigs[0].suptitle('Under WD metric', size=14)
path = os.path.join('..', 'experiments', 'results', 'natural-wd-targeted.pickle')
plot_matrix(ax, path)
ax = subfigs[1].add_subplot()
subfigs[1].suptitle('Under l2 metric', size=14)
path = os.path.join('..', 'experiments', 'results', 'natural-l2-targeted.pickle')
plot_matrix(ax, path)
fig.savefig('figures/targeted_matrix.png', bbox_inches='tight')