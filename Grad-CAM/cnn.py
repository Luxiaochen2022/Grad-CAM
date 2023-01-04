# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:48:04 2022

@author: vipuser
"""
import json
import os
import timeit
from collections import Counter
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import layers, models, Input
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


filename=r'D:\data_test.npz'

data=np.load(filename)
X_waveform = data['X_waveform']
label=data['y']
x_train=X_waveform[0:20000]
y_train=label[0:20000]
x_test=X_waveform[20000:]
y_test=label[20000:]

def model_cnn(input_shape = (2000, 1, 3),
                kernel_size = (3, 1),
                pooling_size = (2, 1),
                root_filters = 16,
                clip_filters = None,
                dense_dropout = 0.3,
                cnn_dropout = 0.1,
                n_layers = 4,
                activation='relu',
                output_class = 3,
                output_activation='softmax'
                ):
    
    # deep learning branch model for comparison purpose. 
    
    # build cnn layers
    inputs = Input(shape=input_shape)

    y = layers.Conv2D(filters=root_filters,
         kernel_size=kernel_size,
         activation=activation, padding="same")(inputs)
    y = layers.MaxPooling2D(pooling_size)(y)
    n_kernels = root_filters
    for i in range(n_layers):

        if clip_filters:
            if n_kernels > clip_filters:
                n_kernels = clip_filters
            else:
                n_kernels *= 2
        else:
            n_kernels *= 2
                
        y = layers.Conv2D(filters=n_kernels,
         kernel_size=kernel_size,padding="same",
                         activation=activation)(y)
        y = layers.Dropout(cnn_dropout)(y)
        y = layers.MaxPooling2D(pooling_size)(y)

    # convert image to vector 
    y = layers.Flatten()(y)
    # dropout regularization
    y = layers.Dropout(dense_dropout)(y)
    y = layers.Dense(100, activation=activation)(y)
    outputs = layers.Dense(output_class, activation=output_activation)(y)
     # model building by supplying inputs/outputs
    baseline_model = models.Model(inputs=inputs, outputs=outputs)
    baseline_model.summary()
    baseline_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['acc'])
    return baseline_model
model=model_cnn()
callbacks = [ModelCheckpoint(filepath=r'D:\cnn1.hdf5', 
                             monitor='val_acc',
                             verbose=1, 
                             save_best_only=True,
                             mode='max'),
             EarlyStopping(monitor='val_acc',
                           mode='max',
                           patience=20)]
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
plt.savefig(r'D:\train1.svg')
