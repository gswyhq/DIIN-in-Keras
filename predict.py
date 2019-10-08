#!/usr/bin/python3
# coding: utf-8

import json
import os
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
    def __init__(self):
        self.model = load_model('models/checkpoint-03-0.49-0.815.hdf5',
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

        sample_inputs[0] = pad_sequences(sample_inputs[0], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # input_word_p
        sample_inputs[1] = pad_sequences(sample_inputs[1], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # input_word_h

        sample_inputs[2] = pad_sequences(sample_inputs[2], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # input_word_p
        sample_inputs[3] = pad_sequences(sample_inputs[3], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # input_word_h

        sample_inputs[6] = pad_sequences(sample_inputs[6], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # exact_match_p
        sample_inputs[7] = pad_sequences(sample_inputs[7], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # exact_match_h

        ret = self.model.predict([sample_inputs])

        return ret

def main():
    model = Predict()
    ret = model.predict('我手机丢了，我想换个手机', '我想买个新手机，求推荐')
    print(ret)

if __name__ == '__main__':
    main()