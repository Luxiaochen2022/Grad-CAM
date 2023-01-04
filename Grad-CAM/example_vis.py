# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:01:26 2022

@author: vipuser
"""

import numpy as np
import cv2
import tensorflow.python.keras as keras
import tensorflow as tf
from matplotlib import pyplot as plt
class GradCAM:
    """Code adopted from Pyimagesearch:
    URL https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    """
    
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        #存储模型，用于度量类的类索引
        # activation map, and the layer to be used when visualizing
        #激活图，以及可视化时要使用的层
        # the class activation map
        #类激活映射
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()
      
    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # 
        #通过反向遍历网络各层，试图找到网络中最后的卷积层。
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
        
    def compute_heatmap(self, image, ps_ratios=None, baseline=False, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            
            (convOutputs, predictions) = gradModel([inputs])

            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        #print(cam.numpy())
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        #denom = (heatmap.max() - heatmap.min()) + eps
        denom = (heatmap.max() - heatmap.min())
        heatmap = numer / denom
        #heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap
    
    def overlay_heatmap(self, heatmap, image, alpha=0.5,
        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
    
mapping = {0:'EQ', 1:"noise",2:'EXP'}









filename=r'D:\paper\data\data_200013.npz'
data=np.load(filename)
X = data['X_waveform']
X=X[20:50]
# y label
y = data['y']

model = keras.models.load_model(r'D:\paper\Model\cnn.hdf5')
y_pred = model.predict(X)

a=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
b=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
for i in a:
    ix = np.array([i])
    sampling_rate = 100
    times = np.arange(2000)/sampling_rate
    cam = GradCAM(model, 0)
    heatmap_eq= cam.compute_heatmap(X[ix])

    cam = GradCAM(model, 1)
    heatmap_exp= cam.compute_heatmap(X[ix])

    cam = GradCAM(model, 2)
    heatmap_noise= cam.compute_heatmap(X[ix])
    
    heatmap_exp /= np.max(heatmap_exp)
    heatmap_eq /= np.max(heatmap_eq)
    heatmap_noise /= np.max(heatmap_noise)
    tr_data = X[ix][0, :, 0, 0]
    y_true = y[ix][0]

    fig, axes = plt.subplots(nrows=3, figsize=(12, 8))

    axes[0].plot(times, tr_data, 'grey')
    ymin, ymax = axes[0].get_ylim()
    sc=axes[0].scatter(times, tr_data, c=heatmap_exp, cmap=plt.cm.jet, zorder=10, 
           s=5, alpha=0.8)

    axes[0].set_title(f'True:  {mapping[y_true]}, Estimate: EQ, with probability {y_pred[ix][0][0]:.2f}')
    axes[0].set_xlim(0, 20)
    axes[0].set_xticks([])

    axes[1].plot(times, tr_data, 'grey')
    sc=axes[1].scatter(times, tr_data, c=heatmap_eq, cmap=plt.cm.jet, zorder=10, 
           s=5, alpha=0.8)
    axes[1].set_title(f'True:  {mapping[y_true]}, Estimate: noise, with probability {y_pred[ix][0][1]:.2f}')
    axes[1].set_xlim(0, 20)
    axes[1].set_xticks([])


    axes[2].plot(times, tr_data, 'grey')
    sc=axes[2].scatter(times, tr_data, c=heatmap_noise, cmap=plt.cm.jet, zorder=10, 
           s=5, alpha=0.8)
    axes[2].set_title(f'True:  {mapping[y_true]}, Estimate: EXP, with probability {y_pred[ix][0][2]:.2f}')
    axes[2].set_xlabel('Time (sec)')
    axes[2].set_xlim(0, 20)
    fig.colorbar(sc,ax=[axes[0],axes[1],axes[2]],label='Normalized Grad-CAM weights')
    path='D:/visualization/'+b[i]+'.png'
    plt.savefig(path)