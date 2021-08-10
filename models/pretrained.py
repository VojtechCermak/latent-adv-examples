import torch
import torch.nn as nn
import numpy as np
import os
import json
import importlib
directory = os.path.dirname(os.path.abspath(__file__))


def import_module(path, name='module'):
    '''
    Imports module from file given its path
    '''
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(path, model_file=None, state_dict=None, device='cuda'):
    '''
    Loads pretrained model from given path.
    '''
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


class PartialGeneratorBase(nn.Module):
    '''
    Base class for partial generators.
    Outputs images with pixels in [0, 1] range.
    '''
    def __init__(self, generator, sections, level, size_z, natural_pixels):
        super().__init__()
        self.generator = generator
        self.sections = sections
        self.level = level
        self.size_z = size_z
        self.natural_pixels = natural_pixels

    def decode(self, x):
        '''
        Decode partial latent vector to image.
        '''
        if self.level == 'full':
            x = self.generator(x)
        else:
            for i in self.sections[self.level]['decode']:
                x = self.generator[i](x)

        if self.natural_pixels:
            return (x + 1) / 2
        else:
            return x

    def encode(self, x):
        '''
        Encode basic (standard normal) latent vector to partial latent vector.
        '''
        if self.level == 'full':
            x = x
        else:
            for i in self.sections[self.level]['encode']:
                x = self.generator[i](x)
        return x


class MnistALI(PartialGeneratorBase):
    '''
    Pretrained MNIST ALI model.
    '''
    def __init__(self, level='full', device='cuda'):
        model = load_model(f'{directory}\\runs\\mnist\\ali-mnist', device=device)
        generator = model.gx.main
        sections = {
            0: {
                'encode': np.arange(0, 3),
                'decode': np.arange(3, len(generator))},
            1: {
                'encode': np.arange(0, 6),
                'decode': np.arange(6, len(generator))},
            2: {
                'encode': np.arange(0, 9),
                'decode': np.arange(9, len(generator))},
            3: {
                'encode': np.arange(0, 12),
                'decode': np.arange(12, len(generator))},
            4: {
                'encode': np.arange(0, 15),
                'decode': np.arange(15, len(generator))},
        }
        super().__init__(generator, sections, level, model.size_z, natural_pixels=True)


class MnistDCGAN(PartialGeneratorBase):
    '''
    Pretrained MNIST DCGAN model.
    '''
    def __init__(self, level='full', device='cuda'):
        model = load_model(f'{directory}\\runs\\mnist\\dcgan-mnist', device=device)
        generator = model.g.main
        sections = {
            0: {
                'encode': np.arange(0, 3),
                'decode': np.arange(3, len(generator))},
            1: {
                'encode': np.arange(0, 6),
                'decode': np.arange(6, len(generator))},
            2: {
                'encode': np.arange(0, 9),
                'decode': np.arange(9, len(generator))},
            3: {
                'encode': np.arange(0, 12),
                'decode': np.arange(12, len(generator))},
        }
        super().__init__(generator, sections, level, model.size_z, natural_pixels=True)


class MnistVAE(PartialGeneratorBase):
    '''
    Pretrained MNIST VAE model.
    '''
    def __init__(self, level='full', device='cuda'):
        model = load_model(f'{directory}\\runs\\mnist\\vae-mnist', device=device)
        generator = model.decoder.main
        sections = {
            0: {
                'encode': np.arange(0, 3),
                'decode': np.arange(3, len(generator))},
            1: {
                'encode': np.arange(0, 6),
                'decode': np.arange(6, len(generator))},
            2: {
                'encode': np.arange(0, 9),
                'decode': np.arange(9, len(generator))},
            3: {
                'encode': np.arange(0, 12),
                'decode': np.arange(12, len(generator))},
        }
        super().__init__(generator, sections, level, model.size_z, natural_pixels=True)


class SvhnALI(PartialGeneratorBase):
    '''
    Pretrained SVHN ALI model.
    '''
    def __init__(self, level='full', device='cuda'):
        model = load_model(f'{directory}\\runs\\svhn\\ali-svhn', device=device)
        generator = model.gx.main
        sections = {
            0: {
                'encode': np.arange(0, 3),
                'decode': np.arange(3, len(generator))},
            1: {
                'encode': np.arange(0, 6),
                'decode': np.arange(6, len(generator))},
            2: {
                'encode': np.arange(0, 9),
                'decode': np.arange(9, len(generator))},
            3: {
                'encode': np.arange(0, 12),
                'decode': np.arange(12, len(generator))},
            4: {
                'encode': np.arange(0, 15),
                'decode': np.arange(15, len(generator))},
            5: {
                'encode': np.arange(0, 18),
                'decode': np.arange(18, len(generator))},
        }
        super().__init__(generator, sections, level, model.size_z, natural_pixels=True)


class SvhnDCGAN(PartialGeneratorBase):
    '''
    Pretrained SVHN DCGAN model.
    '''
    def __init__(self, level='full', device='cuda'):
        model = load_model(f'{directory}\\runs\\svhn\\dcgan-svhn', device=device)
        generator = model.g.main
        sections = {
            0: {
                'encode': np.arange(0, 3),
                'decode': np.arange(3, len(generator))},
            1: {
                'encode': np.arange(0, 6),
                'decode': np.arange(6, len(generator))},
            2: {
                'encode': np.arange(0, 9),
                'decode': np.arange(9, len(generator))},
            3: {
                'encode': np.arange(0, 12),
                'decode': np.arange(12, len(generator))},
        }

        super().__init__(generator, sections, level, model.size_z, natural_pixels=True)


class SvhnVAE(PartialGeneratorBase):
    '''
    Pretrained SVHN VAE model.
    '''
    def __init__(self, level='full', device='cuda'):
        model = load_model(f'{directory}\\runs\\svhn\\vae-svhn', device=device)
        generator = model.decoder.main
        sections = {
            0: {
                'encode': np.arange(0, 3),
                'decode': np.arange(3, len(generator))},
            1: {
                'encode': np.arange(0, 6),
                'decode': np.arange(6, len(generator))},
            2: {
                'encode': np.arange(0, 9),
                'decode': np.arange(9, len(generator))},
            3: {
                'encode': np.arange(0, 12),
                'decode': np.arange(12, len(generator))},
        }
        super().__init__(generator, sections, level, model.size_z, natural_pixels=True)


class CifarALI(PartialGeneratorBase):
    '''
    Pretrained CIFAR ALI model.
    '''
    def __init__(self, level='full', device='cuda'):
        model = load_model(f'{directory}\\runs\\cifar10\\ali-cifar', device=device)
        generator = model.gx.main
        sections = {
            0: {
                'encode': np.arange(0, 3),
                'decode': np.arange(3, len(generator))},
            1: {
                'encode': np.arange(0, 6),
                'decode': np.arange(6, len(generator))},
            2: {
                'encode': np.arange(0, 9),
                'decode': np.arange(9, len(generator))},
            3: {
                'encode': np.arange(0, 12),
                'decode': np.arange(12, len(generator))},
            4: {
                'encode': np.arange(0, 15),
                'decode': np.arange(15, len(generator))},
            5: {
                'encode': np.arange(0, 18),
                'decode': np.arange(18, len(generator))},
        }
        super().__init__(generator, sections, level, model.size_z, natural_pixels=True)


class CifarDCGAN(PartialGeneratorBase):
    '''
    Pretrained CIFAR DCGAN model.
    '''
    def __init__(self, level='full', device='cuda'):
        model = load_model(f'{directory}\\runs\\cifar10\\dcgan-cifar', device=device)
        generator = model.g.main
        sections = {
            0: {
                'encode': np.arange(0, 3),
                'decode': np.arange(3, len(generator))},
            1: {
                'encode': np.arange(0, 6),
                'decode': np.arange(6, len(generator))},
            2: {
                'encode': np.arange(0, 9),
                'decode': np.arange(9, len(generator))},
            3: {
                'encode': np.arange(0, 12),
                'decode': np.arange(12, len(generator))},
        }

        super().__init__(generator, sections, level, model.size_z, natural_pixels=True)



class ClassifierBase(nn.Module):
    '''
    Converts classifier to one that accepts images with pixels in [0, 1] range.
    '''
    def __init__(self, classifier, natural_pixels):
        super().__init__()
        self.classifier = classifier
        self.natural_pixels = natural_pixels

    def forward(self, x):
        if self.natural_pixels:
            return self.classifier(x*2 - 1)
        else:
            return self.classifier(x)


class MnistMadryRobust(ClassifierBase):
    '''
    Madry's classifier with l-inf adversarial training.
    '''
    def __init__(self, device='cuda'):
        classifier = load_model(f'{directory}\\runs\mnist\\pretrained', 'mnist.py', 'mnist_adv.pth', device=device)
        super().__init__(classifier, natural_pixels=True)


class MnistMadryNatural(ClassifierBase):
    '''
    Madry's natural classifier (simple Le-Net architecture).
    '''
    def __init__(self, device='cuda'):
        classifier = load_model(f'{directory}\\runs\mnist\\pretrained', 'mnist.py', 'mnist_natural.pth', device=device)
        super().__init__(classifier, natural_pixels=True)


class MnistClassifier(ClassifierBase):
    '''
    VGG based MNIST classifier.
    '''
    def __init__(self, device='cuda'):
        classifier = load_model(f'{directory}\\runs\\mnist\\cls-mnist', device=device)
        super().__init__(classifier, natural_pixels=True)


class SvhnClassifier(ClassifierBase):
    '''
    VGG based SVHN classifier.
    '''
    def __init__(self, device='cuda'):
        classifier = load_model(f'{directory}\\runs\\svhn\\cls-svhn', device=device)
        super().__init__(classifier, natural_pixels=True)


class CifarClassifier(ClassifierBase):
    '''
    VGG based SVHN classifier.
    '''
    def __init__(self, device='cuda'):
        classifier = load_model(f'{directory}\\runs\\cifar10\\cls-cifar', device=device)
        super().__init__(classifier, natural_pixels=True)