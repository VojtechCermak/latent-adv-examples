'''
Encode arbitrary folder with images to BigBiGAN latent vector z.
Images are rescaled to 256x256.
'''

import os
import argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms

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
        valid_images = [".jpg",".gif",".png",".tga"]
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
        img = Image.open(path)
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
    args = parser.parse_args()

    module = hub.Module('https://tfhub.dev/deepmind/bigbigan-resnet50/1')

    # initialize Tensorflow
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5), # to [-1, 1] pixels
    ])

    print('Loading images')
    dataset = ImageDataset(args.folder, transform)
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    imgs = next(iter(dataloader))

    print('Encoding images')
    z = encode(imgs, sess)
    np.save(os.path.join(args.folder, 'bigbigan_z'), z)

    print('Done')