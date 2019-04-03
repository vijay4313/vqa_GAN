
import sys
import tensorflow as tf  # pylint: disable=protected-access
import numpy as np
import os
import matplotlib.pyplot as plt
sys.path.append("/media/venkatraman/D/thesis/code/utils")
sys.path.append("/media/venkatraman/D/thesis/code/models")
from question_embeddings import QuestionProcessor
from dataHandler import TFDataLoaderUtil
from main import train as genGraph

EPOCH = 2  # The model epoch to test
NUM_SAMPLES = 10  # NUmber of sample images to Demo
VOCAB_SIZE = 1004
EMBED_SIZE = 512
IMG_SHAPE = (448, 448, 3)
CHECKPOINT_ROOT = '/media/venkatraman/D/thesis/model/GAN'


def getCheckpointPath(epoch=None):
    if epoch is None:
        return os.path.abspath(CHECKPOINT_ROOT + "weights")
    else:
        return os.path.abspath(CHECKPOINT_ROOT + "weights_{}".format(epoch))


def getQuesEncoder():
    trainData = TFDataLoaderUtil('/media/venkatraman/D/thesis/data',
                                 'train2014')
    trainQuestions = [value['question']
                      for key, value in trainData.dataset.items()]
    quesEncoder = QuestionProcessor()
    MAX_QUES_PAD_LEN = max(list(map(
                                lambda x: len(quesEncoder.split_sentence(x)),
                                trainQuestions)))
    quesEncoder.generate_vocabulary(trainQuestions)

    return (quesEncoder, MAX_QUES_PAD_LEN)


def main():
    testData = TFDataLoaderUtil('/media/venkatraman/D/thesis/data', 'val2014')
    sampleBatchIds = np.random.choice(list(testData.dataset.keys()),
                                      NUM_SAMPLES,
                                      replace=False)
    sampleTriplets = testData.getQuesImageCompTriplets(sampleBatchIds)

    sampleQues, sampleImgs, sampleCompImgs = testData.dataLoaderFromDataIds(
        sampleTriplets, IMG_SHAPE)
    quesEncoder, MAX_QUES_PAD_LEN = getQuesEncoder()
    sampleQuesEncs = quesEncoder.batch_question_to_token_indices(sampleQues)
    sampleQuesEncPadded = quesEncoder.batch_questions_to_matrix(
        sampleQuesEncs, MAX_QUES_PAD_LEN)

    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    quesVec = tf.placeholder('float32', [None, MAX_QUES_PAD_LEN])
    posImg = tf.placeholder('float32', [None, ]+list(IMG_SHAPE))
    negImg = tf.placeholder('float32', [None, ]+list(IMG_SHAPE))

    discOptimizer, discLoss, genOptimizer, genLoss, genImgData = genGraph(
        quesVec, posImg, negImg, VOCAB_SIZE,
        EMBED_SIZE, MAX_QUES_PAD_LEN, IMG_SHAPE)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, getCheckpointPath(epoch=EPOCH))

    genResult = sess.run(genImgData,
                         feed_dict={quesVec: sampleQuesEncPadded})

    print(genResult.shape)
    print(genResult)

    plt.figure(1)
    for ndx, image in enumerate(genResult):
        plt.subplot(genResult.shape[0], 3, 3*ndx+1)
        plt.imshow(np.interp(image,
                             (image.min(), image.max()),
                             (0, 1)),
                   interpolation="none")
        plt.subplot(genResult.shape[0], 3, 3*ndx+2)
        plt.imshow((255 * sampleImgs[ndx]).astype(np.uint8),
                   cmap="gray",
                   interpolation="none")
        plt.subplot(genResult.shape[0], 3, 3*ndx+3)
        plt.imshow((255 * sampleCompImgs[ndx]).astype(np.uint8),
                   cmap="gray",
                   interpolation="none")
    plt.show()


if __name__ == "__main__":
    main()
