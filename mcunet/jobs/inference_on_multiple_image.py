"""label_image for tflite."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import time
import numpy as np
import os
import torch
from PIL import Image
import tflite_runtime.interpreter as tflite
import torchvision.transforms as T
from mcunet.utils import get_hardware_usage


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_image_location_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def inference_one_image():
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

    dir_path = os.path.dirname(os.path.realpath(args.image))
    target_result = dir_path[-9:]

    results = np.squeeze(output_data)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(args.label_file)

    # print_topk_result(results)
    # print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    print(target_result == labels[top_k[0]][:9])
    print(args.model_file)


def print_topk_result(results):
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(args.label_file)
    for i in top_k:
        print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))


if __name__ == '__main__':
    # Set argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', default=None, help='image to be classified')
    parser.add_argument('-m', '--model_file', default=None, help='.tflite model to be executed')
    parser.add_argument('-l', '--label_file',
                        default='/home/ytwanghaoyu/PycharmProjects/tinyml/mcunet/dataset/imagenet/labels.txt',
                        help='name of file containing labels')
    parser.add_argument('--input_mean', default=127.5, type=float, help='input_mean')
    parser.add_argument('--input_std', default=127.5, type=float, help='input standard deviation')
    parser.add_argument('--num_threads', default=None, type=int, help='number of threads')
    parser.add_argument('--dataset', default='imagenet', type=str)
    parser.add_argument('--val-dir', default='../dataset/imagenet/val', help='path to validation data')
    parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers')
    args = parser.parse_args()

    image_location_list = load_image_location_list('/home/ytwanghaoyu/PycharmProjects/tinyml/mcunet/jobs/img_list.txt')
    image_location_list_len = len(image_location_list)
    network_location_list = [
        '/home/ytwanghaoyu/PycharmProjects/tinyml/mcunet/assets/tflite/mcunet-512kb-2mb_imagenet.tflite',
        '/home/ytwanghaoyu/PycharmProjects/tinyml/mcunet/assets/tflite/mcunet-320kb-1mb_imagenet.tflite']
    args.model_file = network_location_list[0]

    for index in range(100):
        args.image = image_location_list[index]
        cpu_percent, memory_percent = get_hardware_usage.get_HWresource_percent()
        if memory_percent < 39:
            print(memory_percent)
            args.model_file = network_location_list[1]
        if memory_percent >= 39:
            print(memory_percent)
            args.model_file = network_location_list[0]
        inference_one_image()
