#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:02:50 2019

@author: venkatraman
"""

#import matplotlib.pyplot as plt
#import numpy as np
#import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers as L
from tensorflow.keras.applications.vgg16 import VGG16

class SiamGan(object):
    def __init__(self):
        pass
    
    def getGenerator(self, CODE_SIZE):

        generator = Sequential()
        generator.add(L.InputLayer([CODE_SIZE],name='gen_input_layer'))
        generator.add(L.Dense(10*16*16, activation='elu', name = 'gen_Dense1'))

        generator.add(L.Reshape((16,16,10), name = 'gen_reshape'))
        generator.add(L.Conv2DTranspose(16,kernel_size=(9,9),activation='elu', name = 'gen_Conv9_1'))
        generator.add(L.Conv2DTranspose(16,kernel_size=(9,9),activation='elu', name = 'gen_Conv9_2'))
        generator.add(L.Conv2DTranspose(16,kernel_size=(9,9),activation='elu', name = 'gen_Conv9_3'))
        generator.add(L.Conv2DTranspose(16,kernel_size=(9,9),activation='elu', name = 'gen_Conv9_4'))
        generator.add(L.UpSampling2D(size=(2,2), name = 'gen_upsample1'))

        generator.add(L.Conv2DTranspose(16,kernel_size=(7,7),activation='elu', name = 'gen_Conv7_1'))
        generator.add(L.Conv2DTranspose(16,kernel_size=(7,7),activation='elu', name = 'gen_Conv7_2'))
        generator.add(L.UpSampling2D(size=(2,2), name = 'gen_upsample2'))

        generator.add(L.Conv2DTranspose(8,kernel_size=(5,5),activation='elu', name = 'gen_Conv5_1'))
        generator.add(L.Conv2DTranspose(8,kernel_size=(5,5),activation='elu', name = 'gen_Conv5_2'))
        generator.add(L.UpSampling2D(size=(2,2), name = 'gen_upsample3'))
        generator.add(L.Conv2DTranspose(8,kernel_size=(3,3),activation='elu', name = 'gen_Conv3_1'))
        generator.add(L.Conv2DTranspose(8,kernel_size=(3,3),activation='elu', name = 'gen_Conv3_2'))

        generator.add(L.Conv2D(8,kernel_size=3,activation='relu', name = 'gen_Conv3_3'))
        generator.add(L.Conv2D(3,kernel_size=3,activation='tanh', name = 'gen_Conv3_4'))

        return generator

    def getDiscriminator(self, IMG_SHAPE):

        input_tensor = L.Input(shape=IMG_SHAPE, name = 'disc_input_layer')
        input_tensor_norm = tf.layers.batch_normalization(input_tensor, axis = -1, name = 'disc_batch_norm')
        vgg16 = VGG16(input_shape=IMG_SHAPE, weights = 'imagenet',
                        include_top = False)
        
        vgg16.trainable = False
        for layer in vgg16.layers:
            layer.name = 'disc_' + layer.name
        
        vgg16Op = vgg16(input_tensor_norm)

        filterReducn = L.AveragePooling2D(name = 'disc_avg_pool')(vgg16Op)
        filterReducn = L.Conv2D(32, kernel_size=(3,3),
                                activation='relu', name='disc_conv_p1')(filterReducn)

        adversaryTop = L.Flatten()(filterReducn)
        adversaryTop = L.Dense(2048, activation='relu', name='disc_dense1')(adversaryTop)
        discriminator = Model(inputs=input_tensor, outputs=adversaryTop)

        return discriminator

    def tripletLoss(self, anchor, positive, negative, alpha = 0.2):
        """
        Implementation of the triplet loss

        Arguments:
        anchor -- the encodings for the anchor images, of shape (None, 128)
        positive -- the encodings for the positive images, of shape (None, 128)
        negative -- the encodings for the negative images, of shape (None, 128)

        Returns:
        (loss, genLoss) -- tuple of real number, value of the loss for discriminator and generator
        """
        # Step 1: Compute the (encoding) distance between the anchor and
        # the positive, you will need to sum over axis=-1
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)),
                                 axis = -1)
        # Step 2: Compute the (encoding) distance between the anchor and
        # the negative, you will need to sum over axis=-1
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)),
                                 axis = -1)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        # Step 4: Take the maximum of basic_loss and 0.0. Sum over the
        # training examples.
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

        genLoss = tf.reduce_mean(pos_dist)

        return (loss, genLoss)

# =============================================================================
discriminator = SiamGan().getDiscriminator((448, 448, 3))
discriminator.summary()
generator = SiamGan().getGenerator(2048)
generator.summary()

# =============================================================================
