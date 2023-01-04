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


def ConvBNRelu(inpt,ch, kernelsz=3, strides=1, padding='same'):
    x = layers.Conv2D(ch, kernelsz, strides=strides, padding=padding)(inpt)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x
    
def InceptionBlk(inpt,ch, strides=1):
    x1 = ConvBNRelu(inpt,ch,kernelsz=1, strides=strides)
    x2_1 = ConvBNRelu(inpt,ch, kernelsz=1, strides=strides)
    x2_2 = ConvBNRelu(x2_1,ch, kernelsz=3, strides=1)
    x3_1 = ConvBNRelu(inpt,ch, kernelsz=1, strides=strides)
    x3_2 = ConvBNRelu(x3_1,ch, kernelsz=5, strides=1)
    x4_1 = tf.keras.layers.MaxPool2D(3, strides=1, padding='same')(inpt)
    x4_2 = ConvBNRelu(x4_1,ch, kernelsz=1, strides=strides)
    x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
    return x

def Inception10(class_nums):
    inpt = layers.Input(shape=(2000,1,3))
    x=ConvBNRelu(inpt,16)
    
    x=InceptionBlk(x,16, strides=2)
    x=InceptionBlk(x,16, strides=1)
    
    x=InceptionBlk(x,32, strides=2)
    x=InceptionBlk(x,32, strides=1)
    
    x=InceptionBlk(x,64, strides=2)
    x=InceptionBlk(x,64, strides=1)
    
    x=InceptionBlk(x,128, strides=2)
    x=InceptionBlk(x,128, strides=1)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(class_nums, activation='softmax')(x)
    model = tf.keras.Model(inputs=inpt, outputs=x)
    return model


class_num = 3
model = Inception10(3)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          metrics=['acc'])
callbacks = [ModelCheckpoint(filepath=r'D:\inc.hdf5', 
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
plt.savefig(r'D:\inc.svg',bbox_inches='tight')
