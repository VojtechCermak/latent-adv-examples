import torch
import torch.nn as nn
import torch.nn.functional as F

class Transform(nn.Module):
    '''
    Base class for transform functions G(z)
    '''
    pass


class Identity(Transform):
    '''
    G(z) = z
    '''
    def forward(self, x):
        return x


class Decoded(Transform):
    '''
    G(z) = D(z)
    '''
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z):
        return self.decoder.decode(z)


class DecodedDistribution(Decoded):
    '''
    Transform z to decoded x such that its components are from (0, 1) and sum(x) = 1.
    G(z) = D(z) / ||D(z)||
    '''
    def forward(self, z):
        x = self.decoder.decode(z)
        # TODO why do we have 1e-8 here?
        x[x<=0] = 1e-8
        return x / x.norm(1, dim=(2, 3), keepdim=True)


class Distance(nn.Module):
    '''
    Base class for distance functions: d(G(a), G(b))
    '''
    def __init__(self, transform=None):
        super().__init__()
        if transform is None:
            self.G = Identity()
        else:
            self.G = transform

class L2(Distance):
    '''
    Euclidean distance
    '''
    def forward(self, a, b):
        return torch.norm((self.G(a) - self.G(b)), 2, dim=tuple(range(1, a.ndim)))


class SquaredL2(Distance):
    '''
    Squared Euclidean distance
    '''
    def forward(self, a, b):
        return torch.sum((self.G(a) - self.G(b))**2, dim=tuple(range(1, a.ndim)))


class GeomLoss(Distance):
    '''
    Calculates distance from the geomloss library between two BCHW samples.
    Final distance is the sum of distances in each channel.
    '''
    def __init__(self, loss_function, transform=None):
        super().__init__(transform)
        self.loss_function = loss_function

    def forward(self, a, b):
        a_transformed, b_transformed = self.G(a), self.G(b)
        a_weights, a_points = self.weighted_point_cloud(a_transformed)
        b_weights, b_points = self.weighted_point_cloud(b_transformed)
        loss = self.loss_function(a_weights, a_points, b_weights, b_points)
        return loss.view(-1, b_transformed.shape[1]).sum(dim=1)

    def weighted_point_cloud(self, x):
        '''
        Converts BCHW sample to 2D weighted point cloud
        '''
        B, C, H, W = x.shape
        weights = x.view(B*C, H*W)
        a, b = torch.meshgrid(torch.arange(0., H)/H, torch.arange(0., W)/W)
        points = torch.stack((a, b), dim=2).view(-1, 2).to(x.device)
        points = points.repeat(B*C, 1, 1) # Repeat points across 
        return weights, points