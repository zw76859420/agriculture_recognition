# -*- coding:utf-8 -*-
# author:zhangwei

import random
import numpy as np
import keras as kr
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense , Conv2D , MaxPooling2D , Input , Reshape
from keras.layers import BatchNormalization , Dropout , regularizers , Flatten , Activation , GlobalAveragePooling2D
from keras.optimizers import Adam , Adadelta , RMSprop , SGD
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from Agricluture_comp.readPath import *


class ModelFace():
    def __init__(self):
        self.nb_calsses = 61
        # self.filepath = '/home/zhangwei/'
        self.model = self.build_model()

    def build_model(self):
        input_data = Input(shape=[256 , 256 , 3])

        conv1 = Conv2D(filters=16 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(input_data)
        conv1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=16 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(conv1)
        conv2 = BatchNormalization()(conv2)
        pool1 = MaxPooling2D(pool_size=[2 ,2] , strides=[2 , 2])(conv2)
        pool1 = Dropout(0.1)(pool1)

        conv3 = Conv2D(filters=32 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(pool1)
        conv3 = BatchNormalization()(conv3)
        conv4 = Conv2D(filters=32 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(conv3)
        conv4 = BatchNormalization()(conv4)
        pool2 = MaxPooling2D(pool_size=[2 , 2] ,strides=[2 , 2])(conv4)
        pool2 = Dropout(0.1)(pool2)

        conv5 = Conv2D(filters=64 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(pool2)
        conv5 = BatchNormalization()(conv5)
        conv6 = Conv2D(filters=64 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(conv5)
        conv6 = BatchNormalization()(conv6)
        pool3 = GlobalAveragePooling2D()(conv6)

        dense1 = Dense(units=128 , activation='relu' , use_bias=True , kernel_initializer='he_normal')(pool3)
        dense1 = Dropout(0.1)(dense1)
        dense2 = Dense(units=256 , activation='relu' , use_bias=True , kernel_initializer='he_normal')(dense1)
        dense2 = Dropout(rate=0.2)(dense2)
        dense3 = Dense(units=self.nb_calsses , use_bias=True , kernel_initializer='he_normal')(dense2)
        pred = Activation(activation='softmax')(dense3)

        model_data = Model(inputs=input_data , outputs=pred)
        # model_data.summary()
        return model_data

    def train(self , batch_size=128 , nb_epoch=1000 , data_augmentation=False):
        sgd = SGD(lr=0.01 , decay=1e-6 , momentum=0.9 , nesterov=True)
        adam = Adam(lr=0.01 , beta_1=0.9 , beta_2=0.999 , decay=0.9)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy' , metrics=['accuracy'])
        data_gen = generate_batch()
        k = 32739 // batch_size
        if not data_augmentation:
            self.model.fit_generator(data_gen , steps_per_epoch=k , epochs=nb_epoch , verbose=1)
        else:
            datagen = ImageDataGenerator(
                featurewise_center=False ,
                samplewise_center=False ,
                featurewise_std_normalization=False ,
                samplewise_std_normalization=False ,
                zca_whitening=False ,
                rotation_range=20 ,
                width_shift_range=0.2 ,
                height_shift_range=0.2 ,
                horizontal_flip=True ,
                vertical_flip=False
            )
            datagen.fit(data_gen[0])
            self.model.fit_generator(data_gen[0] , data_gen[1])
        self.save_model()


    def save_model(self , filepath='/home/zhangwei/myface_01.model.h5'):
        self.model.save(filepath=filepath)

    def load_mdoel(self , filepath='/home/zhangwei/face/myface.model.h5'):
        self.model = load_model(filepath=filepath)

    def Evaluate(self , dataset):
        score = self.model.evaluate(dataset.valid_images , dataset.valid_labels , verbose=1)
        print("%s:%.2f%%" % (self.model.metrics_names[1] , score[1] * 100))

if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    set_session(tf.Session(config=config))

    filepath = '/home/zhangwei/PycharmProjects/ASR_MFCC/Agricluture_comp/agri_data_01.txt'
    md = ModelFace()
    md.train()