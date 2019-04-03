#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:02:50 2019

@author: venkatraman
"""

# import matplotlib.pyplot as plt
# import numpy as np
# import keras
import tensorflow as tf
# import h5py
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import layers as L
from tensorflow.python.keras.applications.vgg16 import VGG16


class SiamGan(object):
    def __init__(self):
        pass

    def getGenerator(self, CODE_SIZE):

        generator = Sequential()
        generator.add(L.InputLayer([CODE_SIZE], name='gen_input_layer'))
        generator.add(L.Dense(10*16*16, activation='elu', name='gen_Dense1'))

        generator.add(L.Reshape((16, 16, 10), name='gen_reshape'))
        generator.add(L.Conv2DTranspose(16, kernel_size=(9, 9),
                                        activation='elu', name='gen_Conv9_1'))
        generator.add(L.Conv2DTranspose(16, kernel_size=(9, 9),
                                        activation='elu', name='gen_Conv9_2'))
        generator.add(L.Conv2DTranspose(16, kernel_size=(9, 9),
                                        activation='elu', name='gen_Conv9_3'))
        generator.add(L.Conv2DTranspose(16, kernel_size=(9, 9),
                                        activation='elu', name='gen_Conv9_4'))
        generator.add(L.UpSampling2D(size=(2, 2), name='gen_upsample1'))

        generator.add(L.Conv2DTranspose(16, kernel_size=(7, 7),
                                        activation='elu', name='gen_Conv7_1'))
        generator.add(L.Conv2DTranspose(16, kernel_size=(7, 7),
                                        activation='elu', name='gen_Conv7_2'))
        generator.add(L.UpSampling2D(size=(2, 2), name='gen_upsample2'))

        generator.add(L.Conv2DTranspose(8, kernel_size=(5, 5),
                                        activation='elu', name='gen_Conv5_1'))
        generator.add(L.Conv2DTranspose(8, kernel_size=(5, 5),
                                        activation='elu', name='gen_Conv5_2'))
        generator.add(L.UpSampling2D(size=(2, 2), name='gen_upsample3'))
        generator.add(L.Conv2DTranspose(8, kernel_size=(3, 3),
                                        activation='elu', name='gen_Conv3_1'))
        generator.add(L.Conv2DTranspose(8, kernel_size=(3, 3),
                                        activation='elu', name='gen_Conv3_2'))

        generator.add(L.Conv2D(8, kernel_size=3,
                               activation='relu', name='gen_Conv3_3'))
        generator.add(L.Conv2D(3, kernel_size=3,
                               activation=None, name='gen_Conv3_4'))

        return generator

    def getDiscriminator(self, IMG_SHAPE):

        discriminator = Sequential()
        discriminator.add(L.InputLayer(input_shape=IMG_SHAPE,
                                       name='disc_input_layer'))
        discriminator.add(L.BatchNormalization(axis=-1,
                                               name='disc_batch_norm'))
        discriminator.add(L.Conv2D(4, kernel_size=(5, 5),
                                   padding='same',
                                   activation='relu', name='disc_Conv5_1'))
        discriminator.add(L.Conv2D(4, kernel_size=(5, 5),
                                   padding='same',
                                   activation='relu', name='disc_Conv5_2'))
        discriminator.add(L.SpatialDropout2D(0.5))
        discriminator.add(L.MaxPooling2D(pool_size=(2, 2),
                                         strides=(2, 2),
                                         name='disc_max_pool_1'))

        discriminator.add(L.Conv2D(4, kernel_size=(3, 3),
                                   padding='same',
                                   activation='relu', name='disc_Conv3_1'))
        discriminator.add(L.Conv2D(4,  kernel_size=(3, 3),
                                   padding='same',
                                   activation='relu', name='disc_Conv3_2'))
        discriminator.add(L.SpatialDropout2D(0.5))
        discriminator.add(L.MaxPooling2D(pool_size=(2, 2),
                                         strides=(2, 2),
                                         name='disc_max_pool_2'))

        discriminator.add(L.Conv2D(8, kernel_size=(3, 3), padding='same',
                                   activation='relu', name='disc_Conv3_3'))
        discriminator.add(L.Conv2D(8, kernel_size=(3, 3), padding='same',
                                   activation='relu', name='disc_Conv3_4'))
        discriminator.add(L.SpatialDropout2D(0.5))
        discriminator.add(L.MaxPooling2D(pool_size=(2, 2),
                                         strides=(2, 2),
                                         name='disc_max_pool_3'))

        discriminator.add(L.Conv2D(16, kernel_size=(3, 3), padding='same',
                                   activation='relu', name='disc_Conv3_5'))
        discriminator.add(L.MaxPooling2D(pool_size=(2, 2),
                                         strides=(2, 2),
                                         name='disc_max_pool_4'))
        discriminator.add(L.Conv2D(32, kernel_size=(3, 3), padding='same',
                                   activation='relu', name='disc_Conv3_6'))
        discriminator.add(L.SpatialDropout2D(0.5))
        discriminator.add(L.AveragePooling2D(pool_size=(2, 2),
                                             strides=(2, 2),
                                             name='disc_max_pool_5'))

        discriminator.add(L.Flatten())
        discriminator.add(L.Dropout(0.5))
        discriminator.add(L.Dense(2048, activation='relu', name='disc_dense1'))

        return discriminator

    def getDiscriminatorV1(self, IMG_SHAPE):
        input_tensor = L.Input(shape=IMG_SHAPE, name='disc_input_layer')
        input_tensor_norm = L.BatchNormalization(axis=-1,
                                                 name='disc_batch_norm')(
                                                     input_tensor)
        vgg16 = VGG16(input_shape=IMG_SHAPE, weights='imagenet',
                      include_top=False)

        vgg16.trainable = False

        vgg16Op = vgg16(input_tensor_norm)

        filterReducn = L.AveragePooling2D(name='disc_avg_pool')(vgg16Op)
        filterReducn = L.Conv2D(32, kernel_size=(3, 3),
                                activation='relu', name='disc_conv_p1')(
                                    filterReducn)

        adversaryTop = L.Flatten()(filterReducn)
        adversaryTop = L.Dense(2048, activation='relu', name='disc_dense1')(
            adversaryTop)
        discriminator = Model(inputs=input_tensor, outputs=adversaryTop)

        return discriminator

    def tripletLoss(self, anchor, positive, negative, alpha=0.2):
        """
        Implementation of the triplet loss

        Arguments:
        anchor -- the encodings for the anchor images, of shape (None, 128)
        positive -- the encodings for the positive images, of shape (None, 128)
        negative -- the encodings for the negative images, of shape (None, 128)

        Returns:
        (loss, genLoss) -- tuple of real number, value of the
                           loss for discriminator and generator
        """
        # Distance between the anchor and the positive
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)),
                                 axis=-1)
        # Distance between the anchor and the negative
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)),
                                 axis=-1)
        # Subtract the two previous distances and add alpha.
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        # Take the maximum of basic_loss and 0.0. Sum over the
        # training examples.
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

        genLoss = tf.reduce_mean(pos_dist)

        return (loss, genLoss)

# =============================================================================


# discriminator = SiamGan().getDiscriminator((448, 448, 3))
# discriminator.summary()
# generator = SiamGan().getGenerator(2048)
# generator.summary()

# =============================================================================
