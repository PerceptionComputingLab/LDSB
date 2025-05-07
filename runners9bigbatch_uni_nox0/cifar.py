import os
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images_from_batch(batch, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, img in enumerate(batch[b'data']):
        img = img.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, f'{i}.png'))

dir = '/home/qiuxingyu/ddim/ddim-generate/exp/datasets/cifar10_test/cifar-10-batches-py/'
for root,dirs,files in os.walk(dir):
    for file in files:
        print(file)
        if '_batch' in file:
            batch = unpickle(os.path.join(root,file))
            save_images_from_batch(batch, '/home/qiuxingyu/ddim/ddim-generate/exp/datasets/cifar10_test/cifar-10-batches-py/output')
