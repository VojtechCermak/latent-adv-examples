import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import importlib

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

def batch_add_lsb(img_batch, add_lsb=0):
    if add_lsb == 0:
        return img_batch
    elif (add_lsb == 1) or (img_batch.shape[1] == 1):
        sample_int = (img_batch*255).type(torch.uint8)
        img_add = torch.remainder(sample_int, 2).type(torch.float)
        return torch.cat((img_batch, img_add))
    elif add_lsb == 2:
        sample_int = (img_batch*255).type(torch.uint8)
        img_add = torch.remainder(sample_int, 2).type(torch.float)
        img_add0 = torch.Tensor.repeat(img_add[:,0:1,:,:], (1,3,1,1))
        img_add1 = torch.Tensor.repeat(img_add[:,1:2,:,:], (1,3,1,1))
        img_add2 = torch.Tensor.repeat(img_add[:,2:3,:,:], (1,3,1,1))        
        return torch.cat((img_batch, img_add0, img_add1, img_add2))
        
def import_module(path, name='module'):
    '''
    Imports module from file given its path
    '''
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_model(path, model_file=None, state_dict=None, device='cuda'):
    if model_file is None:
        with open(path + '\\' + 'args.json') as f:
            args = json.load(f)
        _, model_file = os.path.split(args['model'])
    if state_dict is None:
        state_dict = 'model_state_dict.pth'

    model = import_module(path + '\\' + model_file).model
    model.load_state_dict(torch.load(path + '\\' + state_dict))
    model = model.to(device)
    model.eval()
    return model


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

        # Collect the predictions of given class
        mask = (class_id == class_target) & (softmax > threshold)
        for tensor in z[mask]:
            data[filled] = tensor
            filled = filled + 1
            if filled >= samples:
                return data

    raise Exception('Not enough samples found. Decrease threshold! ')


###### Sample grid
def sample_grid(classifier, generator, device, no_classes=10):
    '''
    For each class sample one example with different class.
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

    labels = torch.tensor(np.repeat(np.arange(no_classes), no_classes-1), dtype=torch.long, device=device)
    return data_a, data_b, labels
