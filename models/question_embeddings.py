#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 03:26:06 2019

@author: venkatraman
"""
import re
import itertools as it
from collections import Counter
import numpy as np
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as L


class QuestionProcessor:
    def __init__(self, vocab: dict = {}):
        self.vocab = {}
        self.PAD = "#PAD#"
        self.UNK = "#UNK#"
        self.START = "#START#"
        self.END = "#END#"

    def split_sentence(self, sentence):
        return list(filter(lambda x: len(x) > 0,
                           re.split('\W+', sentence.lower())))

    def generate_vocabulary(self, train_captions):
        """
        Return {token: index} for all train tokens (words) that
        occur 5 times or more,
            `index` should be from 0 to N, where N is a number of unique
            tokens in the resulting dictionary.
        Use `split_sentence` function to split sentence into tokens.
        Also, add PAD (for batch padding), UNK (unknown, out of vocabulary),
            START (start of sentence) and END (end of sentence) tokens
            into the vocabulary.
        """

        allWordsTokenSplit = list(it.chain.from_iterable(
                map(lambda x: self.split_sentence(x),
                    train_captions)))

        allWordsTokenDict = Counter(allWordsTokenSplit)
        vocab = list(dict(allWordsTokenDict.most_common(1000)).keys())
        vocab.extend([self.PAD, self.UNK, self.START, self.END])
        self.vocab = {token: index for index, token in enumerate(sorted(vocab))}

    def index_generator_per_question(self, question):
        """
        Return the indices for each token in a single question
        """
        word_list = self.split_sentence(question)

        if self.vocab != {}:
            vocab_keys = set(list(self.vocab.keys()))
            tokenized_question = [self.vocab[self.START]]
            tokenized_question.extend([self.vocab[item] if item in vocab_keys
                                       else self.vocab[self.UNK] for
                                       item in word_list])

            tokenized_question.append(self.vocab[self.END])
            return tokenized_question
        else:
            raise Exception('First initialize the vocabulary to \
                            generate a BoW representation')

    def batch_question_to_token_indices(self, questions):
        """
        Use `split_sentence` function to split sentence into tokens. Replace all
        tokens with vocabulary indices, use UNK for unknown words (out of
        vocabulary). Add START and END tokens to start and end of each sentence
        respectively. For the example above you should produce the following:
            [   [vocab[START], vocab["question1word1"], vocab["question1word2"],
                vocab[END]],
                [vocab[START], vocab["question2word1"], vocab["question2word2"],
                 vocab[END]],
            ... ]
        """
        return [self.index_generator_per_question(question) for question in questions]

    def batch_questions_to_matrix(self, batch_captions, max_len=None):
        """
        `batch_captions` is an array of arrays:
        [
            [vocab[START], ..., vocab[END]],
            [vocab[START], ..., vocab[END]],
            ...
        ]
        Put vocabulary indexed captions into np.array of shape
        (len(batch_captions), columns),
            where "columns" is max(map(len, batch_captions)) when max_len is None
            and "columns" = min(max_len, max(map(len, batch_captions))) otherwise.
        Add padding with pad_idx where necessary.
        Input example: [[1, 2, 3], [4, 5]]
        Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=None
        Output example: np.array([[1, 2], [4, 5]]) if max_len=2
        Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=100
        Try to use numpy, we need this function to be fast!
        """
        if max_len is None:
            max_len = max(map(len, batch_captions))

        temp_matrix = []
        for caption in batch_captions:
            if len(caption) < max_len:
                np_caption = np.array(caption)
                temp_matrix.append(np.pad(np_caption,
                                          (0, max_len - len(caption)),
                                          'constant',
                                          constant_values=self.vocab[self.PAD]))
            else:
                temp_matrix.append(np.array(caption)[:max_len])
        matrix = np.array(temp_matrix)
        return matrix


class QuestionEmbedding(object):
    def __init__(self):
        pass

    def stackedLSTMWordEmbedding(self, vocab_size, embed_size, INP_SIZE,
                                 WORD_EMBED_BOOL=True,
                                 LSTM_UNITS=512, OP_UNITS=2048):
        model = Sequential()
        model.add(L.InputLayer([INP_SIZE], name='gen_Embed_input'))
        if WORD_EMBED_BOOL:
            model.add(L.Embedding(vocab_size, embed_size,
                                  name='gen_Embed_embeddings'))
        model.add(L.LSTM(LSTM_UNITS, return_sequences=True,
                         name='gen_Embed_LSTM_1'))
        model.add(L.LSTM(LSTM_UNITS, return_sequences=False,
                         name='gen_Embed_LSTM_2'))
        model.add(L.Dense(OP_UNITS, activation='elu',
                          name='gen_Embed_Dense_1'))

        return model
