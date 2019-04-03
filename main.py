# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:06:24 2019

@author: venkatraman
"""
import sys
import os
from tqdm import tqdm
import tensorflow as tf  # pylint: disable=protected-access
import numpy as np
sys.path.append("/media/venkatraman/D/thesis/code/utils")
sys.path.append("/media/venkatraman/D/thesis/code/models")

from question_embeddings import QuestionProcessor, QuestionEmbedding
from dataHandler import TFDataLoaderUtil
from siam_gan import SiamGan

VOCAB_SIZE = 1004
BATCH_SIZE = 16
EMBED_SIZE = 512
IMG_SHAPE = (448, 448, 3)
NUM_EPOCHS = 100
CHECKPOINT_ROOT = '/media/venkatraman/D/thesis/model/GAN'


def getCheckpointPath(epoch=None):
    if epoch is None:
        return os.path.abspath(CHECKPOINT_ROOT + "weights")
    else:
        return os.path.abspath(CHECKPOINT_ROOT + "weights_{}".format(epoch))


def train(quesVec, posImg, negImg, VOCAB_SIZE,
          EMBED_SIZE, QUES_SIZE, IMG_SHAPE,
          DISC_LR=1e-4, GEN_LR=1e-3, GEN_BETA1=0.9, GEN_BETA2=0.999):

    # Generate Question Embeddings
    quesEmbeds = QuestionEmbedding().stackedLSTMWordEmbedding(
        vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, INP_SIZE=QUES_SIZE)

    siamGAN = SiamGan()

    posImgEmbeds = siamGAN.getDiscriminator(IMG_SHAPE)(posImg)

    negImgEmbeds = siamGAN.getDiscriminator(IMG_SHAPE)(negImg)

    genImgdata = siamGAN.getGenerator(2048)(quesEmbeds(quesVec))

    genImgEmbeds = siamGAN.getDiscriminator(IMG_SHAPE)(genImgdata)

    discLoss, genLoss = siamGAN.tripletLoss(genImgEmbeds,
                                            posImgEmbeds,
                                            negImgEmbeds)
    # regularize

    discOptimizer = tf.train.GradientDescentOptimizer(
                                                    DISC_LR).minimize(discLoss)

    genOptimizer = tf.train.AdamOptimizer(
        learning_rate=GEN_LR,
        beta1=GEN_BETA1,
        beta2=GEN_BETA2).minimize(genLoss)

    return (discOptimizer, discLoss, genOptimizer, genLoss, genImgdata)


def run():
    trainData = TFDataLoaderUtil('/media/venkatraman/D/thesis/data',
                                 'train2014')
    trainBatches = trainData.genDataBatchesIds(BATCH_SIZE=BATCH_SIZE)
    # print(trainData.taskType)
    # print(trainData.dataset)
    # print(list(trainData.dataset))
    trainQuestions = [value['question']
                      for key, value in trainData.dataset.items()]

    quesEncoder = QuestionProcessor()
    MAX_QUES_PAD_LEN = max(list(map(
                                lambda x: len(quesEncoder.split_sentence(x)),
                                trainQuestions)))
    print(MAX_QUES_PAD_LEN)
    quesEncoder.generate_vocabulary(trainQuestions)

    valData = TFDataLoaderUtil('/media/venkatraman/D/thesis/data', 'val2014')
    valBatches = valData.genDataBatchesIds(BATCH_SIZE=BATCH_SIZE)

    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    quesVec = tf.placeholder('float32', [None, MAX_QUES_PAD_LEN])
    posImg = tf.placeholder('float32', [None, ]+list(IMG_SHAPE))
    negImg = tf.placeholder('float32', [None, ]+list(IMG_SHAPE))
    discOptimizer, discLoss, genOptimizer, genLoss, _ = train(
        quesVec, posImg, negImg, VOCAB_SIZE,
        EMBED_SIZE, MAX_QUES_PAD_LEN, IMG_SHAPE)

    # intialize all variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    start = 0
    # saver.restore(s, get_checkpoint_path(epoch=start))

    for epoch in tqdm(range(start, NUM_EPOCHS)):
        _trainDiscLoss = 0.0
        _trainGenLoss = 0.0
        _valGenLoss = 0.0
        _valDiscLoss = 0.0
        counter = 0
        # subValBatches = np.random.choice(valBatches, 2048, replace=False)

        for trainBatch in tqdm(trainBatches):
            # print(len(trainBatch))
            batchTriplets = trainData.getQuesImageCompTriplets(trainBatch)
            # print(len(batchTriplets))
            batchQues, batchImgs, batchCompImgs = trainData.dataLoaderFromDataIds(
                batchTriplets, IMG_SHAPE)
            batchQuesEncodings = quesEncoder.batch_question_to_token_indices(batchQues)
            batchQuesEncPadded = quesEncoder.batch_questions_to_matrix(
                batchQuesEncodings, MAX_QUES_PAD_LEN)
            # print(batchQuesEncPadded.shape)
            # print(batchQuesEncPadded[:2, :])
            # print(batchImgs.shape)
            # print(batchCompImgs.shape)

            feedDict = {
                quesVec: batchQuesEncPadded,
                posImg: batchImgs,
                negImg: batchCompImgs
            }

            for i in range(5):
                _trainDiscLoss += sess.run([discOptimizer, discLoss],
                                           feed_dict=feedDict)[1]

            _trainGenLoss += sess.run([genOptimizer, genLoss],
                                      feed_dict=feedDict)[1]

        _trainDiscLoss /= float(len(trainBatches))
        _trainGenLoss /= float(len(trainBatches))

        print('='*50)
        print('Epoch: ' + str(epoch))
        print('train: Disc loss: {}, Gen loss: {}'.format(_trainDiscLoss,
                                                          _trainGenLoss))

        for valBatch in valBatches:
            batchTriplets = valData.getQuesImageCompTriplets(valBatch)
            batchQues, batchImgs, batchCompImgs = valData.dataLoaderFromDataIds(
                batchTriplets, IMG_SHAPE)
            batchQuesEncodings = quesEncoder.batch_question_to_token_indices(batchQues)
            batchQuesEncPadded = quesEncoder.batch_questions_to_matrix(
                batchQuesEncodings, MAX_QUES_PAD_LEN)

            feedDict = {
                quesVec: batchQuesEncPadded,
                posImg: batchImgs,
                negImg: batchCompImgs
            }

            _valDiscLoss += sess.run(discLoss, feed_dict=feedDict)

            _valGenLoss += sess.run(genLoss, feed_dict=feedDict)

        _valDiscLoss /= float(len(valBatches))
        _valGenLoss /= float(len(valBatches))

        print('val: Disc loss: {}, Gen loss: {}'.format(_valDiscLoss,
                                                        _valGenLoss))

        saver.save(sess, getCheckpointPath(epoch))

    saver.save(sess, getCheckpointPath('final'))
    print('*'*50)
    print('Finished')


if __name__ == "__main__":
    run()
