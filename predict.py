#!/usr/bin/python3
# coding: utf-8

import json
import os
import sys
import numpy as np
from keras.models import load_model
from feature_extractors.densenet import DenseNet
from layers.decaying_dropout import DecayingDropout
from layers.encoding import Encoding
from layers.interaction import Interaction
from lcqmc_preprocess import LCQMCPreprocessor
from keras.preprocessing.sequence import pad_sequences
from train import DIIN

WORD_VECTORS_FILE = 'data/word-vectors.npy'

class Predict():
    def __init__(self, model_path='models/checkpoint-03-0.47-0.809.hdf5'):
        self.model = load_model(model_path,
                           custom_objects={'DIIN': DIIN, 'DecayingDropout': DecayingDropout, 'Encoding': Encoding,
                                           'Interaction': Interaction, 'DenseNet': DenseNet})
        self.lcqmc_preprocessor = LCQMCPreprocessor()

        word2id_path = os.path.splitext(WORD_VECTORS_FILE)[0] + '-word2id.json'
        with open(word2id_path, 'r')as f:
            word_to_id = json.load(f)
        part_of_speech_to_id_path = os.path.splitext(WORD_VECTORS_FILE)[0] + '-speech2id.json'
        with open(part_of_speech_to_id_path, 'r')as f:
            part_of_speech_to_id = json.load(f)

        self.lcqmc_preprocessor.word_to_id = word_to_id
        self.lcqmc_preprocessor.part_of_speech_to_id = part_of_speech_to_id


    def predict(self, premise, hypothesis, max_words_p=32, max_words_h=32, chars_per_word=16):

        sample_inputs = self.lcqmc_preprocessor.parse_one(premise, hypothesis,
                                       max_words_h=max_words_h, max_words_p=max_words_p,
                                       chars_per_word=chars_per_word)
        sample_result = list(sample_inputs)
        # print('sample_inputs: {}'.format(sample_inputs))
        res = [[], [], [], [], [], [], [], []]

        for res_item, parsed_item in zip(res, sample_result):
            res_item.append(parsed_item)
        res[0] = pad_sequences(res[0], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # input_word_p
        res[1] = pad_sequences(res[1], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # input_word_h

        res[2] = pad_sequences(res[2], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # input_word_p
        res[3] = pad_sequences(res[3], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # input_word_h

        res[6] = pad_sequences(res[6], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # exact_match_p
        res[7] = pad_sequences(res[7], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # exact_match_h
        sample_inputs = [np.array(r) for r in res]

        ret = self.model.predict(sample_inputs)

        return ret

def main():

    model = Predict(model_path='models/checkpoint-03-0.47-0.809.hdf5')
    if len(sys.argv) > 1:
        premise, hypothesis = sys.argv[1:3]
    else:
        premise, hypothesis = '我手机丢了，我想换个手机', '我想买个新手机，求推荐'
    ret = model.predict(premise, hypothesis)

    print('`{}`与`{}`，预测结果：{}，及其概率：{}'.format(premise, hypothesis, ret.argmax(), ret.max()))

if __name__ == '__main__':
    main()