# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:55:10 2022

@author: vipuser
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt
from tensorflow.keras import layers
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

filename=r'D:\data_test.npz'

data=np.load(filename)
X_waveform = data['X_waveform']
x_train=X_waveform[0:20000]
label=data['y']
y_train=label[0:20000]


def conv2d_bn(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same'):
    '''卷积、归一化和relu三合一'''
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(inpt)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def basic_bottle(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False):
    '''18中的4个basic_bottle'''
    x = conv2d_bn(inpt, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    residual=inpt
    if if_baisc==True:
        residual = conv2d_bn(inpt, filters=filters, kernel_size=(1,1), strides=2, padding='same')
        residual = layers.BatchNormalization()(residual)
        out = layers.add([x, residual])
        out = tf.keras.layers.Activation('relu')(out)
    else:
        out = layers.add([x, residual])
        out = tf.keras.layers.Activation('relu')(out)
    return out

def resnet18(class_nums):
    '''主模型'''
    inpt = layers.Input(shape=(2000,1,3))
    #layer 1
    x = conv2d_bn(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same')
    #layer 2
    x = basic_bottle(x, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    x = basic_bottle(x, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    #layer 3
    x = basic_bottle(x, filters=128, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=128, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 4
    x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 5
    x = basic_bottle(x, filters=512, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=512, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    #GlobalAveragePool
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(class_nums, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())(x)
    model = tf.keras.Model(inputs=inpt, outputs=x)
    return model


class_num = 3
model = resnet18(class_num)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          metrics=['acc'])
callbacks = [ModelCheckpoint(filepath=r'D:\resnet.hdf5', 
                             monitor='val_acc',
                             verbose=1, 
                             save_best_only=True,
                             mode='max'),
             EarlyStopping(monitor='val_acc',
                           mode='max',
                           patience=30)]
model.summary()

history=model.fit(x_train, y_train, epochs=50, 
                    validation_split=0.2,
                    batch_size=128,
                    shuffle=True,
                    callbacks=callbacks,
                                )
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
plt.savefig(r'D:\resnet.svg',bbox_inches='tight')