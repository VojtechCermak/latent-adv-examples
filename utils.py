import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def fix_seed(seed=1):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def grid_plot(img_batch, save_as=None, nrows=6, already_grid=False, figsize=None):
    '''
    Plot tensor batch of images. Input dimension format: (B,C,W,H)
    '''
    if already_grid:
        grid = img_batch
    else:
        grid = make_grid(img_batch, nrow=nrows, normalize=True)
    grid = grid.cpu().detach().numpy()
    grid = np.transpose(grid, (1,2,0))
    if figsize == None:
        figsize = (nrows, nrows)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(grid)
    if save_as is not None:
        fig.savefig(save_as)
        plt.close(fig)
    else:
        plt.show()


def class_sampler(classifier, generator, class_target, samples=100, threshold=0.99, max_steps=200, batch_size=32, device='cuda'):
    '''
    Sample data from single class, labeled by auxiliary classifier.
        classifier: Auxiliary classifier used for label assignment.
        generator: Generator used for sampling.
        class_target: Class id which will be sampled.
        threshold: Threshold of softmax needed for sample to be considered as given class.
    '''
    filled = 0
    data = torch.zeros((samples, generator.size_z, 1, 1), device=device)
    for i in range(max_steps):
        # Make predictions
        with torch.no_grad():
            z = torch.randn(batch_size, generator.size_z, 1, 1, device=device)
            imgs = generator.decode(z)
            output = F.softmax(classifier(imgs), 1).data.cpu()
            softmax, class_id = output.max(dim=1)

        # Collect predictions of given class
        mask = (class_id == class_target) & (softmax > threshold)
        for tensor in z[mask]:
            data[filled] = tensor
            filled = filled + 1
            if filled >= samples:
                return data

    raise Exception('Not enough samples found. Decrease threshold! ')


def sample_grid(classifier, generator, device, no_classes=10):
    '''
    Sample 10 x 9 grid of class pairs.
    '''
    data_a = []
    data_b = []
    for a in range(no_classes):
        data_a.append(class_sampler(classifier, generator, a, samples=no_classes-1, threshold=0.99, device=device))
        for b in range(no_classes):
            if a != b:
                data_b.append(class_sampler(classifier, generator, b, samples=1, threshold=0.99, device=device))
    data_a = torch.cat(data_a)
    data_b = torch.cat(data_b)
    return data_a, data_b