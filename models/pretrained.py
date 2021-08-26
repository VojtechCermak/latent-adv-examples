import torch
import torch.nn as nn
from torchvision import transforms
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
        with open(os.path.join(path, 'args.json')) as f:
            args = json.load(f)
        _, model_file = os.path.split(args['model'])
    if state_dict is None:
        state_dict = 'model_state_dict.pth'

    model = import_module(os.path.join(path, model_file)).model
    model.load_state_dict(torch.load(os.path.join(path, state_dict)))
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
        model = load_model(os.path.join(directory, 'runs', 'mnist', 'ali-mnist'), device=device)
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
        model = load_model(os.path.join(directory, 'runs', 'mnist', 'dcgan-mnist'), device=device)
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
        model = load_model(os.path.join(directory, 'runs', 'mnist', 'vae-mnist'), device=device)
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
        model = load_model(os.path.join(directory, 'runs', 'svhn', 'ali-svhn'), device=device)
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
        model = load_model(os.path.join(directory, 'runs', 'svhn', 'dcgan-svhn'), device=device)
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
        model = load_model(os.path.join(directory, 'runs', 'svhn', 'vae-svhn'), device=device)
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
        model = load_model(os.path.join(directory, 'runs', 'cifar10', 'ali-cifar'), device=device)
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
        model = load_model(os.path.join(directory, 'runs', 'cifar10', 'dcgan-cifar'), device=device)
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


class ImagenetBigBiGAN():
    def __init__(self, level, device='cuda'):
        from pytorch_pretrained_gans import make_gan
        model = make_gan(gan_type='bigbigan')
        model = model.eval().to(device)
        self.model = model
        self.generator = model.big_gan
        self.level = level

    def __call__(self, z):
        x = self.model(z)
        return (x + 1) / 2

    def prepare(self, z):
        classes = torch.zeros(z.shape[0], dtype=torch.int64, device=z.device)
        y = self.generator.shared(classes)

        # If hierarchical, concatenate zs and ys
        if self.generator.hier:
            zs = torch.split(z, self.generator.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.generator.blocks)
        return ys, z

    def encode(self, z):
        '''
        Caveat: due to hierarchical structure of z, some parts of z are stored in self.ys attribute.
        As results, decode can be used only on the currently encoded z.
        '''
        # Prepare
        self.ys, h = self.prepare(z)

        # First linear layer
        h = self.generator.linear(h)
        h = h.view(h.size(0), -1, self.generator.bottom_width, self.generator.bottom_width)

        for index, blocklist in enumerate(self.generator.blocks[:self.level]):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, self.ys[index])
        return h

    def decode(self, h):
        for index, blocklist in enumerate(self.generator.blocks[self.level:]):
            # Second inner loop in case block has multiple layers
            index = index + self.level
            for block in blocklist:
                h = block(h, self.ys[index])
        x = torch.tanh(self.generator.output_layer(h))
        return (x + 1) / 2


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
        classifier = load_model(os.path.join(directory, 'runs', 'mnist', 'pretrained'), 'mnist.py', 'mnist_adv.pth', device=device)
        super().__init__(classifier, natural_pixels=True)


class MnistMadryNatural(ClassifierBase):
    '''
    Madry's natural classifier (simple Le-Net architecture).
    '''
    def __init__(self, device='cuda'):
        classifier = load_model(os.path.join(directory, 'runs', 'mnist', 'pretrained'), 'mnist.py', 'mnist_natural.pth', device=device)
        super().__init__(classifier, natural_pixels=True)


class MnistClassifier(ClassifierBase):
    '''
    VGG based MNIST classifier.
    '''
    def __init__(self, device='cuda'):
        classifier = load_model(os.path.join(directory, 'runs', 'mnist', 'cls-mnist'), device=device)
        super().__init__(classifier, natural_pixels=True)


class SvhnClassifier(ClassifierBase):
    '''
    VGG based SVHN classifier.
    '''
    def __init__(self, device='cuda'):
        classifier = load_model(os.path.join(directory, 'runs', 'svhn', 'cls-svhn'), device=device)
        super().__init__(classifier, natural_pixels=True)


class CifarClassifier(ClassifierBase):
    '''
    VGG based SVHN classifier.
    '''
    def __init__(self, device='cuda'):
        classifier = load_model(os.path.join(directory, 'runs', 'cifar10', 'cls-cifar'), device=device)
        super().__init__(classifier, natural_pixels=True)


class EfficientNetClassifier(nn.Module):
    '''
    Pretrained Efficient net classifier.
    '''
    def __init__(self, device='cuda'):
        from efficientnet_pytorch import EfficientNet
        super().__init__()
        self.classifier = EfficientNet.from_pretrained('efficientnet-b0')
        self.classifier = self.classifier.eval().to(device)

    def forward(self, x):
        tfs = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std  = [0.229, 0.224, 0.225]),
            ])
        return self.classifier(tfs(x))
