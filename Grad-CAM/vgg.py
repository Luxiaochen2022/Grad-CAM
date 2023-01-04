# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:24:02 2022

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

def VGG16(num_classes):
    inpt = layers.Input(shape=(2000,1,3))
    
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(inpt)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(filters=128,kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inpt, outputs=x)
    return model
model = VGG16(3)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          metrics=['acc'])
callbacks = [ModelCheckpoint(filepath=r'D:\vgg.hdf5', 
                             monitor='val_acc',
                             verbose=1, 
                             save_best_only=True,
                             mode='max'),
             EarlyStopping(monitor='val_acc',
                           mode='max',
                           patience=20)]
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
plt.savefig(r'D:\vgg.svg',bbox_inches='tight')