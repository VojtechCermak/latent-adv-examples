'''
Encode arbitrary folder with images to BigBiGAN latent vector z.
Images are rescaled to 256x256.
'''

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from torchvision.utils import make_grid, save_image

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
tf.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')

class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.imgs = self.find_imgs(folder)

    def find_imgs(self, folder):
        imgs = []
        valid_images = [".jpg",".gif",".png",".tga", ".jpeg"]
        for f in os.listdir(folder):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            else:
                imgs.append(os.path.join(folder, f))
        return imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    
def encode(x, sess):
    x = x.permute(0, 2, 3, 1).numpy()
    images = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    encode = module(images, signature='encode', as_dict=True)
    z = sess.run(encode, feed_dict={images: x})['z_sample']
    return z

def decode(z, sess):
    latent = tf.placeholder(tf.float32, shape=[None, 120])
    decode = module(latent, signature='generate')
    imgs = sess.run(decode, feed_dict={latent: z})
    return torch.tensor(imgs).permute(0, 3, 1, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest="folder", help = 'path to folder with images', required=True)
    parser.add_argument('-batch_size', dest="batch_size", type=int, default=16)
    parser.add_argument('-model', dest="model", default="resnet")
    args = parser.parse_args()

    # Create output folder
    folder = os.path.join(args.folder, 'encoded')
    if not os.path.exists(folder):
        os.makedirs(folder)

    print('Loading BigBiGAN model')
    if args.model == "resnet":
        module = hub.Module('https://tfhub.dev/deepmind/bigbigan-resnet50/1')
    elif args.model == "revnet":
        module = hub.Module('https://tfhub.dev/deepmind/bigbigan-revnet50x4/1')
    else:
        raise ValueError('Invalid model')

    print('Initializing BigBiGAN model')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    images = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    latent = tf.placeholder(tf.float32, shape=[None, 120])
    encode = module(images, signature='encode', as_dict=True)
    decode = module(latent, signature='generate')

    print('Loading images')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5), # to [-1, 1] pixels
    ])
    dataset = ImageDataset(args.folder, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    for i, imgs in enumerate(dataloader):

        print(f'Encoding batch {i}')
        z = sess.run(encode, feed_dict={images: imgs.permute(0, 2, 3, 1).numpy()})['z_sample']
        np.save(os.path.join(folder, f'encoded_z_{i}'), z)

        imgs_rec = sess.run(decode, feed_dict={latent: z})
        imgs_rec = torch.tensor(imgs_rec).permute(0, 3, 1, 2)

        result = torch.cat([F.avg_pool2d(imgs, 2), imgs_rec])
        grid = make_grid(result, nrow=imgs.shape[0], normalize=True)
        save_image(grid, os.path.join(folder, f'encoded_img_{i}.png'))

    print('Done')