"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

import torchvision.transforms as T


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
    # Set argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='/home/ytwanghaoyu/PycharmProjects/tinyml/mcunet/dataset/imagenet/val/n01644373/ILSVRC2012_val_00034542.JPEG',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='/home/ytwanghaoyu/PycharmProjects/tinyml/mcunet/assets/tflite/mcunet-256kb-1mb_imagenet.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/home/ytwanghaoyu/PycharmProjects/tinyml/mcunet/dataset/imagenet/labels.txt',
        help='name of file containing labels')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    parser.add_argument(
        '--num_threads', default=None, type=int, help='number of threads')
    args = parser.parse_args()

    interpreter = tflite.Interpreter(
        model_path=args.model_file, num_threads=args.num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ''' Method 1 : NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]  # 160
    width = input_details[0]['shape'][2]  # 160
    image = Image.open(args.image).resize((width, height))
    image = T.ToTensor()(image)
    # image is a float32 tensor array as <class 'torch.Tensor'>
    # it start with ([[[ with a torch.Size([3, height, width])
    # '''

    # ''' Method 2 : Use resolution to get a square size as height and do normalization
    resolution = input_details[0]['shape'][1]
    image = T.Compose([
        T.Resize(int(resolution * 256 / 224)),
        T.CenterCrop(resolution),
        T.ToTensor(),
    ])(Image.open(args.image))
    # image is a float32 tensor array as <class 'torch.Tensor'>
    # it start with ([[[ with a torch.Size([3, resolution, resolution])
    # '''

    # add N dim
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # add a new dims
        # input_data = np.expand_dims(img, axis=0)  # suitable for np.array/ same as above code
    # print(image.shape)  # ([[[[ torch.Size([1, 3, resolution, resolution])
    image = image.permute(0, 2, 3, 1)  # change dims position from (0,1,2,3) to (0,2,3,1)
    input_data = image.cpu().numpy()  # change into <class 'numpy.ndarray'>
    input_data = (input_data * 255 - 128).astype(np.int8)

    # Set as input data
    interpreter.set_tensor(input_details[0]['index'], np.int8(input_data))

    # Start inference and save time cost
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    results = np.squeeze(output_data)
    # print(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(args.label_file)
    for i in top_k:
        print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

'''

import sys

sys.path.append(".")
import argparse
import torch.backends.cudnn as cudnn
import os
import numpy as np
from torchvision import datasets, transforms
import numpy as np
import tensorflow as tf
import torch
from mcunet.utils import accuracy

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(
    model_path="/home/ytwanghaoyu/PycharmProjects/tinyml/mcunet/assets/tflite/mcunet-320kb-1mb_imagenet.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
resolution = input_shape[1]

# ------------------start caching the test set

val_dir = '../dataset/imagenet/val'
val_dataset = datasets.ImageFolder(val_dir, transform=transforms.Compose([
    transforms.Resize(int(resolution * 256 / 224)),  # range [0, 1]
    transforms.CenterCrop(resolution),
    transforms.ToTensor(),
]))
val_loader = torch.utils.data.DataLoader(
    val_dataset, shuffle=False)
val_loader_cache = [v for v in val_loader]
images = torch.cat([v[0] for v in val_loader_cache], dim=0)
targets = torch.cat([v[1] for v in val_loader_cache], dim=0)

val_loader_cache = [[x, y] for x, y in zip(images, targets)]
print('done.')
print('dataset size:', len(val_loader_cache))

image, target = val_loader_cache[1]

if len(image.shape) == 3:
    image = image.unsqueeze(0)
image = image.permute(0, 2, 3, 1)
image_np = image.cpu().numpy()
image_np = (image_np * 255 - 128).astype(np.int8)
# --------------------------------------------------


input_data = np.array(np.random.random_sample(input_shape), dtype=np.int8)
interpreter.set_tensor(input_details[0]['index'], image_np.reshape(*input_shape))

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])  # get a <class 'numpy.ndarray'>
output = torch.from_numpy(output_data).view(1, -1)  # get a <class 'torch.Tensor'>
acc1, acc5 = accuracy(output, target.view(1), topk=(1, 5))  # acc1, acc5 is <class 'torch.Tensor'>
print(acc1.item())
'''
