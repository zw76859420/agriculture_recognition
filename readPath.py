# -*- coding:utf-8 -*-
# author:zhangwei

import numpy as np
import cv2
import os
import random
from keras.utils import np_utils

width_shape = 256
heigh_shape = 256
dir_name = '/home/zhangwei/data/AgriculturalDisease_trainingset/images/'

def resize_image(image , height=width_shape , width=heigh_shape):
    top , bottom , left , right = (0 , 0 , 0 , 0)
    h , w , channels = image.shape
    # print(h , w)
    longest_edge = max(h , w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
        # print(dw , left , right)
    else:
        pass
    # print(top , bottom , left , right)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    return cv2.resize(constant, (height, width))

# def preprocess(inputs):
#     # imputs = np.array(inputs , dtype=np.float)
#     inputs = inputs / 255.
#     inputs -= 0.5
#     inputs *= 2.
#     return inputs

def read_path(labelpath , batch_size=128):
    with open(labelpath , 'r') as fr:
        lines = fr.readlines()
        data_num = len(lines)
        images = np.zeros(shape=[data_num , 256 , 256 , 3] , dtype=np.float32)
        labels = np.zeros(shape=[data_num] , dtype=np.int32)
        for i in range(data_num):
            path = lines[i].strip().split('?')[1]
            label = lines[i].strip().split('?')[0]
            labels[i] = label
            image = cv2.imread(dir_name + path)
            # image = preprocess(image)
            image = resize_image(image)
            images[i] = image
        # i = 0
        # for line in lines:
        #     i += 1
        #     res = line.strip().split('?')
        #     label = res[0]
        #     labels.append(label)
        #     image = cv2.imread(dir_name + res[1])
        #     image = np.array(image , dtype=np.float)
        #     image = preprocess(image)
        #     image = resize_image(image , 256 , 256)
        #     images.append(image)
        #     print(image)
        return images , labels

def get_data(n_start , labelpath='/home/zhangwei/PycharmProjects/ASR_MFCC/Agricluture_comp/agri_data_01.txt'):
    with open(labelpath) as fr:
        lines = fr.readlines()
        path = lines[n_start].strip().split('?')[1]
        label = lines[n_start].strip().split('?')[0]
        label = np_utils.to_categorical(label , 61)
        filepath = dir_name + path
        img = cv2.imread(filepath)
        img = resize_image(img)
        return img , label

def batch_data(labelpath='/home/zhangwei/PycharmProjects/ASR_MFCC/Agricluture_comp/agri_data_01.txt' , batch_size=128):
    with open(labelpath , 'r') as fr:
        lines = fr.readlines()
        num = len(lines)
        images = []
        labels = []
        for i in range(batch_size):
            n = random.randint(0 , num-1)
            image , label = get_data(n)
            images.append(image)
            labels.append(label)
        images = np.array(images , dtype=np.float)
        labels = np.array(labels , dtype=np.int)
        return images , labels

def generate_batch():
    while True:
        images , labels = batch_data()
        yield images , labels


if __name__ == '__main__':
    label_data_path = '/home/zhangwei/PycharmProjects/ASR_MFCC/Agricluture_comp/agri_data_01.txt'
    a = generate_batch()
    for i ,j in a:
        print(j)