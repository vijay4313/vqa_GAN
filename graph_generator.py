# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rgb VAE - GAN implementation for CloudML."""

import argparse
import os

import logging
import tensorflow as tf

from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils

from siam_gan import SiamGan
from question_embeddings import QuestionEmbedding, QuestionProcessor
from dataHandler import TFDataLoaderUtil

# Global constants
TRAIN, EVAL = 'TRAIN', 'EVAL'
PREDICT_IMAGE_IN = 'PREDICT_IMAGE_IN'



def override_if_not_in_args(flag, argument, args):
  """Checks if flags is in args, and if not it adds the flag to args."""
  if flag not in args:
    args.extend([flag, argument])


def build_signature(inputs, outputs):
  """Build the signature for use when exporting the graph.

  Args:
    inputs: a dictionary from tensor name to tensor
    outputs: a dictionary from tensor name to tensor
  Returns:
    The signature, a SignatureDef proto, specifies the input/output tensors
    to bind when running prediction.
  """
  signature_inputs = {
      key: saved_model_utils.build_tensor_info(tensor)
      for key, tensor in inputs.items()
  }
  signature_outputs = {
      key: saved_model_utils.build_tensor_info(tensor)
      for key, tensor in outputs.items()
  }

  signature_def = signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.PREDICT_METHOD_NAME)

  return signature_def


def create_model():
  """Factory method that creates model to be used by generic task.py."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--DISC_LR', type=float, default=1e-4)
  parser.add_argument('--GEN_LR', type=float, default=1e-3)
  parser.add_argument('--GEN_BETA1', type=float, default=0.9)
  parser.add_argument('--GEN_BETA2', type=float, default=0.999)
  parser.add_argument('--IMAGE_SIZE', type=int, default=None)
  parser.add_argument('--QUES_SIZE', type=int, default=None)
  parser.add_argument('--QUES_EMBED_SIZE', type=int, default=2048)
  parser.add_argument('--WORD_EMBED_SIZE', type=int, default=512)
  parser.add_argument('--VOCAB_SIZE', type=int, default=1004)
  args, task_args = parser.parse_known_args()
  override_if_not_in_args('--max_steps', '1000', task_args)
  override_if_not_in_args('--batch_size', '64', task_args)
  override_if_not_in_args('--eval_set_size', '370', task_args)
  override_if_not_in_args('--eval_interval_secs', '2', task_args)
  override_if_not_in_args('--log_interval_secs', '2', task_args)
  override_if_not_in_args('--min_train_eval_rate', '2', task_args)

  return Model(args.DISC_LR, args.GEN_LR, args.GEN_BETA1, args.GEN_BETA2,
               args.IMAGE_SIZE, args.QUES_EMBED_SIZE, args.WORD_EMBED_SIZE,
               args.QUES_SIZE, args.VOCAB_SIZE), task_args


class GraphReferences(object):
  """Holder of base tensors used for training model using common task."""

  def __init__(self):
    self.posImg = None
    self.negImg = None
    self.quesVec = None
    self.global_step = None
    self.posImgEmbeds = None
    self.negImgEmbeds = None
    self.genImgdata = None
    self.genImgEmbeds = None
    self.discLoss = None
    self.genLoss = None
    self.discOptimizer = None
    self.genOptimizer = None
    self.keys = None


class Model(object):
  """Tensorflow model for Rgb VAE-GAN."""

  def __init__(self, DISC_LR, GEN_LR, GEN_BETA1, GEN_BETA2,
               IMAGE_SIZE, QUES_EMBED_SIZE, WORD_EMBED_SIZE,
               QUES_SIZE, VOCAB_SIZE):
    """Initializes VAE-GAN. DCGAN architecture: https://arxiv.org/abs/1511.06434

    Args:
      learning_rate: The learning rate for the three networks.
      dropout: The dropout rate for training the network.
      beta1: Exponential decay rate for the 1st moment estimates.
      resized_image_size: Desired size of resized image.
      crop_image_dimension: Square size of the bounding box.
      center_crop: True iff images should be center cropped.
    """
    self.DISC_LR = DISC_LR
    self.GEN_LR = GEN_LR
    self.GEN_BETA1 = GEN_BETA1
    self.GEN_BETA2 = GEN_BETA2
    self.IMAGE_SIZE = IMAGE_SIZE
    self.QUES_EMBED_SIZE = QUES_EMBED_SIZE
    self.WORD_EMBED_SIZE = WORD_EMBED_SIZE
    self.VOCAB_SIZE = VOCAB_SIZE
    self.QUES_SIZE = QUES_SIZE
    self.has_exported_embed_in = False
    self.has_exported_image_in = False
    self.batch_size = 0
    self.tokenizer = QuestionProcessor()
    self.trainTFDataset = None
    self.evalTFDataset = None
    self.MAX_QUES_PAD_LEN = 100
    
    
  def leaky_relu(self, x, name, leak=0.2):
    """Leaky relu activation function.

    Args:
      x: input into layer.
      name: name scope of layer.
      leak: slope that provides non-zero y when x < 0.

    Returns:
      The leaky relu activation.
    """
    return tf.maximum(x, leak * x, name=name)

  def build_graph(self, data_dir, batch_size, mode):
    """Builds the VAE-GAN network.

    Args:
      batch_size: Batch size of input data.
      mode: Mode of the graph (TRAINING, EVAL, or PREDICT)

    Returns:
      The tensors used in training the model.
    """
    tensors = GraphReferences()
    assert batch_size > 0
    self.batch_size = batch_size
    
    if mode in (TRAIN, EVAL):
        trainData = TFDataLoaderUtil(data_dir, 'train2014')
        trainQuestions = [value['question'] for key, value in trainData.dataset.items()]
        self.MAX_QUES_PAD_LEN = max(list(map(lambda x: len(self.tokenizer.split_sentence(x)), trainQuestions)))
        self.tokenizer.generate_vocabulary(trainQuestions)
        logging.info('Size of Question Vectors: %d', self.MAX_QUES_PAD_LEN)
        
        self.trainTFDataset = trainData.genTFDatasetObject(self.tokenizer, 
                                                           self.MAX_QUES_PAD_LEN, 
                                                           self.batch_size, 
                                                           self.NUM_PARALLEL_CALLS, 
                                                           self.BUFFER_SIZE)
        tensors.quesVec = self.trainTFDataset.get_next()[0]
        tensors.posImg = self.trainTFDataset.get_next()[1]
        tensors.negImg = self.trainTFDataset.get_next()[2]
    
    if mode is EVAL:
        evalData = TFDataLoaderUtil(data_dir, 'val2014')        
        self.evalTFDataset = evalData.genTFDatasetObject(self.tokenizer, 
                                                         self.MAX_QUES_PAD_LEN, 
                                                         self.batch_size, 
                                                         self.NUM_PARALLEL_CALLS, 
                                                         self.BUFFER_SIZE)
        
        tensors.quesVec = self.evalTFDataset.get_next()[0]
        tensors.posImg = self.evalTFDataset.get_next()[1]
        tensors.negImg = self.evalTFDataset.get_next()[2]    
    
    siamGAN = SiamGan()
    quesEmbeds = QuestionEmbedding().stackedLSTMWordEmbedding(
            vocab_size=self.VOCAB_SIZE, 
            embed_size=self.WORD_EMBED_SIZE, 
            INP_SIZE=self.QUES_SIZE)
    
    tensors.posImgEmbeds = siamGAN.getDiscriminator(self.IMG_SHAPE)(tensors.posImage)
    
    tensors.negImgEmbeds = siamGAN.getDiscriminator(self.IMG_SHAPE)(tensors.negImg)

    tensors.genImgdata = siamGAN.getGenerator(self.QUES_EMBED_SIZE)(quesEmbeds(self.quesVec))

    tensors.genImgEmbeds = siamGAN.getDiscriminator(self.IMG_SHAPE)(tensors.genImgdata)
    
    
    if mode in (EVAL, TRAIN):

        tensors.discLoss, tensors.genLoss = siamGAN.tripletLoss(
                tensors.genImgEmbeds, 
                tensors.posImgEmbeds, 
                tensors.negImgEmbeds)
        #regularize
    
        tf.summary.scalar('cost_generator', tensors.genLoss)
        tf.summary.scalar('cost_discriminator', tensors.discLoss)
        tf.summary.tensor_summary('disc_pos', tensors.posImgEmbeds)
        tf.summary.tensor_summary('disc_neg', tensors.negImgEmbeds)
        tf.summary.scalar('mean_disc_pos', tf.reduce_mean(tensors.posImgEmbeds))
        tf.summary.scalar('mean_disc_neg', tf.reduce_mean(tensors.negImgEmbeds))
    
        # Cost of Decoder/Generator is VAE network cost and cost of generator
        # being detected by the discriminator.
        tensors.global_step = tf.Variable(0, name='global_step', trainable=False)
        t_vars = tf.trainable_variables()
    
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
          generator_vars = [var for var in t_vars if var.name.startswith('gen_')]
          discriminator_vars = [
              var for var in t_vars if var.name.startswith('disc_')
          ]
          
          tensors.discOptimizer =  tf.train.GradientDescentOptimizer(
                  self.DISC_LR).minimize(
                          tensors.discLoss,
                          var_list = discriminator_vars,
                          global_step = tensors.global_step)
          
          tensors.genOptimizer = tf.train.AdamOptimizer(
                  learning_rate = self.GEN_LR, 
                  beta1 = self.GEN_BETA1, 
                  beta2 = self.GEN_BETA2).minimize(
                          tensors.genLoss,
                          var_list = generator_vars,
                          global_step = tensors.global_step)

    return tensors

  def build_train_graph(self, data_dir, batch_size):
    """Builds the training VAE-GAN graph.

    Args:
      data_paths: Locations of input data.
      batch_size: Batch size of input data.

    Returns:
      The tensors used in training the model.
    """
    return self.build_graph(data_dir, batch_size, mode=TRAIN)

  def build_eval_graph(self, data_dir, batch_size):
    """Builds the evaluation VAE-GAN graph.

    Args:
      data_paths: Locations of input data.
      batch_size: Batch size of input data.

    Returns:
      The tensors used in training the model.
    """
    return self.build_graph(data_dir, batch_size, mode=EVAL)

  def build_prediction_image_graph(self):
    """Builds the prediction VAE-GAN graph for image input.

    Returns:
      The inputs and outputs of the prediction.
    """
    tensors = self.build_graph(1, PREDICT_IMAGE_IN)

    keys_p = tf.placeholder('float32', [None, self.QUES_SIZE])
    inputs = {'key': keys_p, 'ques_vector': tensors.quesVec}
    keys = tf.identity(keys_p)
    outputs = {'key': keys, 'prediction': tensors.genImgData}

    return inputs, outputs

  def export(self, last_checkpoint, output_dir):
    """Exports the prediction graph.

    Args:
      last_checkpoint: The last checkpoint saved.
      output_dir: Directory to save graph.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        inputs, outputs = self.build_prediction_image_graph()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        trained_saver = tf.train.Saver()
        trained_saver.restore(sess, last_checkpoint)
    
        predict_signature_def = build_signature(inputs, outputs)
        # Create a saver for writing SavedModel training checkpoints.
        build = builder.SavedModelBuilder(
            os.path.join(output_dir, 'saved_model_image_in'))
        build.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    predict_signature_def
            },
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
        self.has_exported_image_in = True
        build.save()
