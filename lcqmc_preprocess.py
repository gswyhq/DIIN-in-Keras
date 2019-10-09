from __future__ import print_function

import argparse
import io
import json
import os
from jieba import posseg
from keras.utils import np_utils
import gensim #导入gensim包
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from util import get_snli_file_path, get_word2vec_file_path, ChunkDataManager
from gensim.models import KeyedVectors

def pad(x, maxlen):
    if len(x) <= maxlen:
        pad_width = ((0, maxlen - len(x)), (0, 0))
        return np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
    res = x[:maxlen]
    return np.array(res, copy=False)


class BasePreprocessor(object):

    def __init__(self):
        self.word_to_id = {}
        self.char_to_id = {}
        self.vectors = []
        self.part_of_speech_to_id = {}
        self.unique_words = set()  # 所有的单词集合
        self.unique_parts_of_speech = set() # 所有单词对应的词性集合

    @staticmethod
    def load_data(file_path):
        """
        Load jsonl file by default
        """
        with open(file_path) as f:
            lines = f.readlines()
            datas = [t.strip().split('\t') for t in lines]
            # datas = datas[:len(datas)//20]
            return [[t[0], t[1], int(t[2])] for t in datas if len(t)==3]

    @staticmethod
    def load_word_vectors(file_path, separator=' ', normalize=True, max_words=None):
        """
        :return: words[], np.array(vectors)
        """
        seen_words = set()
        words = []
        vectors = []

        print('Loading', file_path)
        if 'Tencent_AILab_ChineseEmbedding' in file_path:
            wv_from_text = KeyedVectors.load_word2vec_format(file_path, binary=False)
            words = wv_from_text.wv.index2word
            vectors = np.array([wv_from_text.get_vector(word) for word in words], dtype='float32', copy=False)
        else:
            model = gensim.models.Word2Vec.load(file_path)  # 加载词向量模型
            words = model.wv.index2word
            vector_size = model.vector_size
            vectors = [model.wv.get_vector(word) for word in words]
            vectors = np.array(vectors, dtype='float32', copy=False)
        return words, vectors

    def get_words_with_part_of_speech(self, sentence):
        """
        :return: words, parts_of_speech
        """
        raise NotImplementedError

    def get_sentences(self, sample):
        """
        :param sample: sample from data
        :return: premise, hypothesis
        """
        raise NotImplementedError

    def get_all_words_with_parts_of_speech(self, file_paths):
        """
        :param file_paths: 数据文件路径
        :return: words, parts_of_speech
        """
        all_words = [] # 所有的数据（训练、验证、测试）中的词组成的列表
        all_parts_of_speech = [] # 所有的数据（训练、验证、测试）中的词对应的词性组成的列表
        # ['/home/gswyhq/data/LCQMC/train.txt',
        # '/home/gswyhq/data/LCQMC/test.txt',
        # '/home/gswyhq/data/LCQMC/dev.txt']
        for file_path in file_paths:
            data = self.load_data(file_path=file_path)
            # data[0]
            # Out[18]:
            # {'annotator_labels': ['neutral',
            #                       'entailment',
            #                       'neutral',
            #                       'neutral',
            #                       'neutral'],
            #  'captionID': '4705552913.jpg#2',
            #  'gold_label': 'neutral',
            #  'pairID': '4705552913.jpg#2r1n',
            #  'sentence1': 'Two women are embracing while holding to go packages.',
            #  'sentence1_binary_parse': '( ( Two women ) ( ( are ( embracing ( while ( holding ( to ( go packages ) ) ) ) ) ) . ) )',
            #  'sentence1_parse': '(ROOT (S (NP (CD Two) (NNS women)) (VP (VBP are) (VP (VBG embracing) (SBAR (IN while) (S (NP (VBG holding)) (VP (TO to) (VP (VB go) (NP (NNS packages)))))))) (. .)))',
            #  'sentence2': 'The sisters are hugging goodbye while holding to go packages after just eating lunch.',
            #  'sentence2_binary_parse': '( ( The sisters ) ( ( are ( ( hugging goodbye ) ( while ( holding ( to ( ( go packages ) ( after ( just ( eating lunch ) ) ) ) ) ) ) ) ) . ) )',
            #  'sentence2_parse': '(ROOT (S (NP (DT The) (NNS sisters)) (VP (VBP are) (VP (VBG hugging) (NP (UH goodbye)) (PP (IN while) (S (VP (VBG holding) (S (VP (TO to) (VP (VB go) (NP (NNS packages)) (PP (IN after) (S (ADVP (RB just)) (VP (VBG eating) (NP (NN lunch))))))))))))) (. .)))'}

            for premise, hypothesis, label in tqdm(data):
                # sentence1_parse, sentence2_parse
                # premise, hypothesis
                # Out[21]:
                # ('(ROOT (S (NP (CD Two) (NNS women)) (VP (VBP are) (VP (VBG embracing) (SBAR (IN while) (S (NP (VBG holding)) (VP (TO to) (VP (VB go) (NP (NNS packages)))))))) (. .)))',
                # '(ROOT (S (NP (DT The) (NNS sisters)) (VP (VBP are) (VP (VBG hugging) (NP (UH goodbye)) (PP (IN while) (S (VP (VBG holding) (S (VP (TO to) (VP (VB go) (NP (NNS packages)) (PP (IN after) (S (ADVP (RB just)) (VP (VBG eating) (NP (NN lunch))))))))))))) (. .)))')
                # premise, hypothesis = self.get_sentences(sample)
                premise_words,    premise_speech    = self.get_words_with_part_of_speech(premise)
                # print(premise_words, premise_speech)
                # ['Two', 'women', 'are', 'embracing', 'while', 'holding', 'to', 'go', 'packages', '.']
                # ['CD', 'NNS', 'VBP', 'VBG', 'IN', 'VBG', 'TO', 'VB', 'NNS', '.']  # 英语文档的词性标记列表
                hypothesis_words, hypothesis_speech = self.get_words_with_part_of_speech(hypothesis)
                # print(hypothesis_words, hypothesis_speech)
                # ['The', 'sisters', 'are', 'hugging', 'goodbye', 'while', 'holding', 'to', 'go', 'packages', 'after','just', 'eating', 'lunch', '.']
                # ['DT', 'NNS', 'VBP', 'VBG', 'UH', 'IN', 'VBG', 'TO', 'VB', 'NNS', 'IN', 'RB', 'VBG', 'NN', '.']
                all_words           += premise_words  + hypothesis_words + list(premise) + list(hypothesis)
                all_parts_of_speech += premise_speech + hypothesis_speech

        self.unique_words           = set(all_words)
        self.unique_parts_of_speech = set(all_parts_of_speech)

    @staticmethod
    def get_not_present_word_vectors(not_present_words, word_vector_size, normalize):
        res_words = []
        res_vectors = []
        for word in not_present_words:
            vec = np.random.uniform(size=word_vector_size)
            if normalize:
                vec /= np.linalg.norm(vec, ord=2)
            res_words.append(word)
            res_vectors.append(vec)
        return res_words, res_vectors

    def init_word_to_vectors(self, vectors_file_path, needed_words, normalize=False, max_loaded_word_vectors=None):
        """
        对语料中的词，映射对应的词向量，及word2id:
            {word -> vec} mapping
            {word -> id}  mapping
            [vectors] array
        :param max_loaded_word_vectors: maximum number of words to load from word-vec file
        :param vectors_file_path: file where word-vectors are stored (Glove .txt file)
        :param needed_words: words for which to keep word-vectors
        :param normalize: normalize word vectors
        """
        needed_words = set(needed_words)
        # 词向量对应的词列表，及对应词向量
        words, self.vectors = self.load_word_vectors(file_path=vectors_file_path,
                                                     normalize=normalize,
                                                     max_words=max_loaded_word_vectors)
        word_vector_size = self.vectors.shape[-1]  # 词向量的维数
        self.vectors = list(self.vectors)

        present_words = needed_words.intersection(words)  # 语料中的词与词向量的交集
        not_present_words = needed_words - present_words  # 没有对应词向量的词列表；
        print('#Present words:', len(present_words), '\t#Not present words', len(not_present_words))

        # 对应缺失词向量的词，随机初始化生成对应的词向量；
        not_present_words, not_present_vectors = self.get_not_present_word_vectors(not_present_words=not_present_words,
                                                                                   word_vector_size=word_vector_size,
                                                                                   normalize=normalize)
        words, self.vectors = zip(*[(word, vec) for word, vec in zip(words, self.vectors) if word in needed_words])
        words = list(words) + not_present_words
        self.vectors = list(self.vectors) + not_present_vectors

        print('Initializing word mappings...')
        self.word_to_id  = {word: i   for i, word   in enumerate(words)}
        self.vectors = np.array(self.vectors, copy=False)

        assert len(self.word_to_id) == len(self.vectors)
        print(len(self.word_to_id), 'words in total are now initialized!')

    def init_chars(self, words):
        """
        获取所有的字集合，及char2id
        """
        chars = set()
        for word in words:
            chars = chars.union(set(word))

        self.char_to_id = {char: i+1 for i, char in enumerate(chars)}
        print('Chars:', chars)

    def init_parts_of_speech(self, parts_of_speech):
        # 词性及其对应的id;
        self.part_of_speech_to_id = {part: i+1 for i, part in enumerate(parts_of_speech)}
        print('Parts of speech:', parts_of_speech)

    def save_word_vectors(self, file_path):
        np.save(file_path, self.vectors)

        word2id_path = os.path.splitext(file_path)[0] + '-word2id.json'
        with open(word2id_path, 'w')as f:
            json.dump(self.word_to_id, f, ensure_ascii=False)

        part_of_speech_to_id_path = os.path.splitext(file_path)[0] + '-speech2id.json'
        with open(part_of_speech_to_id_path, 'w')as f:
            json.dump(self.part_of_speech_to_id, f, ensure_ascii=False)

    def get_label(self, sample):
        return NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def label_to_one_hot(self, label):
        label_set = self.get_labels()
        # print('label_set, {}'.format(label_set))
        res = np.zeros(shape=(len(label_set)), dtype=np.bool)
        i = label_set.index(label)
        res[i] = 1
        return res

    def parse_sentence(self, sentence, max_words, chars_per_word):
        """
        解析一个句子，返回对应的词列表、词性列表、词id列表、词性id列表、词性填充补全矩阵；字填充补全矩阵；
        :param sentence: 输出的句子
        :param max_words: 可处理的最大词数
        :param chars_per_word: 每个词最大的字符数；
        :return: 
        """
        # Words
        words, parts_of_speech = self.get_words_with_part_of_speech(sentence)
        # ['Two', 'women', 'are', 'embracing', 'while', 'holding', 'to', 'go', 'packages', '.']
        # ['CD', 'NNS', 'VBP', 'VBG', 'IN', 'VBG', 'TO', 'VB', 'NNS', '.']
        word_ids = [self.word_to_id[word] for word in words]
        # print('words: {}, parts_of_speech: {}, word_ids: {}'.format(words, parts_of_speech, word_ids))

        # Syntactical features
        syntactical_features = [self.part_of_speech_to_id[part] for part in parts_of_speech]
        syntactical_one_hot = np.eye(len(self.part_of_speech_to_id) + 2)[syntactical_features]  # 将词性转换成0-1向量；

        # Chars
        # chars = [[self.char_to_id[c] for c in word] for word in words]
        # chars = pad_sequences(chars, maxlen=chars_per_word, padding='post', truncating='post')  # 不足长度在尾部补0，超长截断
        char_ids = [self.word_to_id[w] for w in sentence]
        # chars.shape
        # Out[60]: (10, 16)
        # syntactical_features
        # Out[63]: [33, 3, 36, 30, 39, 30, 29, 11, 3, 41]
        # syntactical_one_hot.shape
        # Out[64]: (10, 44)
        # pad(syntactical_one_hot, max_words).shape
        # Out[61]: (32, 44)
        # pad(chars, max_words).shape
        # Out[65]: (32, 16)

        # return (words, parts_of_speech, np.array(word_ids, copy=False),
        #         syntactical_features, pad(syntactical_one_hot, max_words),
        #         pad(chars, max_words))

        return (words, parts_of_speech, np.array(word_ids, copy=False),
                syntactical_features, pad(syntactical_one_hot, max_words),
                np.array(char_ids, copy=False))

    def parse_one(self, premise, hypothesis, max_words_p, max_words_h, chars_per_word):
        """
        :param premise: sentence
        :param hypothesis: sentence
        :param max_words_p: maximum number of words in premise
        :param max_words_h: maximum number of words in hypothesis
        :param chars_per_word: number of chars in each word
        :return: (premise_word_ids, hypothesis_word_ids,
                  premise_char_ids, hypothesis_char_ids,
                  premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                  premise_exact_match, hypothesis_exact_match)
        """
        # 词列表、词性列表、词id列表、词性id列表、词性填充补全矩阵；字填充补全矩阵；
        (premise_words, premise_parts_of_speech, premise_word_ids,
         premise_syntactical_features, premise_syntactical_one_hot,
         premise_char_ids) = self.parse_sentence(sentence=premise, max_words=max_words_p, chars_per_word=chars_per_word)

        (hypothesis_words, hypothesis_parts_of_speech, hypothesis_word_ids,
         hypothesis_syntactical_features, hypothesis_syntactical_one_hot,
         hypothesis_char_ids) = self.parse_sentence(sentence=hypothesis, max_words=max_words_h, chars_per_word=chars_per_word)

        def calculate_exact_match(source_words, target_words):
            source_words = [word.lower() for word in source_words]
            target_words = [word.lower() for word in target_words]
            target_words = set(target_words)

            res = [(word in target_words) for word in source_words]
            return np.array(res, copy=False)

        # 对应词在对方中是否存在
        premise_exact_match    = calculate_exact_match(premise_words, hypothesis_words)
        # array([False, False,  True, False,  True,  True,  True,  True,  True, True])
        hypothesis_exact_match = calculate_exact_match(hypothesis_words, premise_words)
        # array([False, False,  True, False, False,  True,  True,  True,  True, True, False, False, False, False,  True])

        return (premise_word_ids, hypothesis_word_ids,
                premise_char_ids, hypothesis_char_ids,
                premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                premise_exact_match, hypothesis_exact_match)

    def parse(self, input_file_path, max_words_p=33, max_words_h=20, chars_per_word=13):
        """
        :param input_file_path: file to parse data from
        :param max_words_p: maximum number of words in premise
        :param max_words_h: maximum number of words in hypothesis
        :param chars_per_word: number of chars in each word (padding is applied if not enough)
        :return: (premise_word_ids, hypothesis_word_ids,
                  premise_char_ids, hypothesis_char_ids,
                  premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                  premise_exact_match, hypothesis_exact_match)
        """
        # res = [premise_word_ids, hypothesis_word_ids, premise_char_ids, hypothesis_char_ids,
        # premise_syntactical_one_hot, hypothesis_syntactical_one_hot, premise_exact_match, hypothesis_exact_match]
        res = [[], [], [], [], [], [], [], [], []]
        # 句1词id列表,句2词id列表,句1字填充补全矩阵,句2字填充补全矩阵,句1词性填充补全矩阵,句2词性填充补全矩阵,句1对应词对方是否包含,句2对应词对方是否包含,标签

        data = self.load_data(input_file_path)  # q1, q2, label
        for premise, hypothesis, label in tqdm(data):
            # As stated in paper: The labels are "entailment", "neutral", "contradiction" and "-".
            # "-"  shows that annotators can't reach consensus with each other, thus removed during training and testing
            # label = self.get_label(sample=sample)  # 可选的标签值有： {'-', 'contradiction', 'entailment', 'neutral'}
            # if label == '-':
            #     continue
            # premise, hypothesis = self.get_sentences(sample=sample)
            # (ROOT (S (NP (CD Two) (NNS women)) (VP (VBP are) (VP (VBG embracing) (SBAR (IN while) (S (NP (VBG holding)) (VP (TO to) (VP (VB go) (NP (NNS packages)))))))) (. .)))
            # (ROOT (S (NP (DT The) (NNS sisters)) (VP (VBP are) (VP (VBG hugging) (NP (UH goodbye)) (PP (IN while) (S (VP (VBG holding) (S (VP (TO to) (VP (VB go) (NP (NNS packages)) (PP (IN after) (S (ADVP (RB just)) (VP (VBG eating) (NP (NN lunch))))))))))))) (. .)))

            sample_inputs = self.parse_one(premise, hypothesis,
                                           max_words_h=max_words_h, max_words_p=max_words_p,
                                           chars_per_word=chars_per_word)
            # 词id列表，字填充补全矩阵，词性填充补全矩阵，对应词对方是否包含

            # label = self.label_to_one_hot(label=label)
            label = list(np_utils.to_categorical(label, num_classes=2))
            sample_result = list(sample_inputs) + [label]
            for res_item, parsed_item in zip(res, sample_result):
                res_item.append(parsed_item)

        res[0] = pad_sequences(res[0], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # input_word_p
        res[1] = pad_sequences(res[1], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # input_word_h

        res[2] = pad_sequences(res[2], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # input_word_p
        res[3] = pad_sequences(res[3], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # input_word_h

        res[6] = pad_sequences(res[6], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # exact_match_p
        res[7] = pad_sequences(res[7], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # exact_match_h
        return res


class LCQMCPreprocessor(BasePreprocessor):
    def get_words_with_part_of_speech(self, sentence):

        words = []
        parts_of_speech = []
        for word, flag in posseg.cut(sentence):
            parts_of_speech.append(flag)
            words.append(word)
        return words, parts_of_speech

    def get_sentences(self, sample):
        return sample['sentence1_parse'], sample['sentence2_parse']

    def get_label(self, sample):
        return sample['gold_label']

    def get_labels(self):
        return [0, 1]


def preprocess(p, h, chars_per_word, preprocessor, save_dir, data_paths,
               word_vector_save_path, normalize_word_vectors, max_loaded_word_vectors=None, word_vectors_load_path=None,
               include_word_vectors=True, include_chars=True,
               include_syntactical_features=True, include_exact_match=True):

    preprocessor.get_all_words_with_parts_of_speech([data_path[1] for data_path in data_paths])
    print('Found', len(preprocessor.unique_words), 'unique words')
    print('Found', len(preprocessor.unique_parts_of_speech), 'unique parts of speech')

    # Init mappings of the preprocessor
    preprocessor.init_word_to_vectors(vectors_file_path=word_vectors_load_path,
                                      needed_words=preprocessor.unique_words,
                                      normalize=normalize_word_vectors,
                                      max_loaded_word_vectors=max_loaded_word_vectors)
    # preprocessor.init_chars(words=preprocessor.unique_words)

    preprocessor.init_parts_of_speech(parts_of_speech=preprocessor.unique_parts_of_speech)

    # 预处理及保存数据；
    preprocessor.save_word_vectors(word_vector_save_path)  # 保存相关词（包括缺失词向量的词随机初始化的词向量）的向量到文件
    for dataset, input_path in data_paths:
        # [('train', '/home/gswyhq/data/LCQMC/train.txt'),
        #  ('test', '/home/gswyhq/data/LCQMC/test.txt'),
        #  ('dev', '/home/gswyhq/data/LCQMC/dev.txt')]
        data = preprocessor.parse(input_file_path=input_path,
                                  max_words_p=p,
                                  max_words_h=h,
                                  chars_per_word=chars_per_word)
        # 句1词id列表,句2词id列表,句1字填充补全矩阵,句2字填充补全矩阵,句1词性填充补全矩阵,句2词性填充补全矩阵,句1对应词对方是否包含,句2对应词对方是否包含,标签

        # Determine which part of data we need to dump
        if not include_exact_match:             del data[6:8]  # Exact match feature
        if not include_syntactical_features:    del data[4:6]  # Syntactical POS tags
        if not include_chars:                   del data[2:4]  # Character features
        if not include_word_vectors:            del data[0:2]  # Word vectors

        data_saver = ChunkDataManager(save_data_path=os.path.join(save_dir, dataset))
        data_saver.save([np.array(item) for item in data])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p',              default=32,         help='Maximum words in premise',            type=int)
    parser.add_argument('--h',              default=32,         help='Maximum words in hypothesis',         type=int)
    parser.add_argument('--chars_per_word', default=16,         help='Number of characters in one word',    type=int)
    parser.add_argument('--max_word_vecs',  default=None,       help='Maximum number of word vectors',      type=int)
    parser.add_argument('--train_data_dir',       default='/home/gswyhq/data/LCQMC',    help='训练语料路径',              type=str)
    parser.add_argument('--save_dir',       default='data/lcqmc/',    help='Save directory of data',              type=str)
    parser.add_argument('--dataset',        default='lcqmc',     help='Which preprocessor to use',           type=str)
    parser.add_argument('--word_vec_load_path', default="/home/gswyhq/data/WordVector_60dimensional/wiki.zh.text.model",   help='Path to load word vectors',           type=str)
    parser.add_argument('--word_vec_save_path', default='data/lcqmc-word-vectors.npy', help='Path to save vectors', type=str)
    parser.add_argument('--normalize_word_vectors',      action='store_true')
    parser.add_argument('--omit_word_vectors',           action='store_true')
    parser.add_argument('--omit_chars',                  action='store_true')
    parser.add_argument('--omit_syntactical_features',   action='store_true')
    parser.add_argument('--omit_exact_match',            action='store_true')
    args = parser.parse_args()

    if args.dataset == 'lcqmc':
        lcqmc_preprocessor = LCQMCPreprocessor()
        path = args.train_data_dir # '/home/gswyhq/data/LCQMC' # /notebooks/data/LCQMC
        train_path = os.path.join(path, 'train.txt')
        test_path  = os.path.join(path, 'test.txt')
        dev_path   = os.path.join(path, 'dev.txt')

        preprocess(p=args.p, h=args.h, chars_per_word=args.chars_per_word,
                   preprocessor=lcqmc_preprocessor,
                   save_dir=args.save_dir,
                   data_paths=[('train', train_path), ('test', test_path), ('dev', dev_path)],
                   word_vectors_load_path=args.word_vec_load_path,
                   normalize_word_vectors=args.normalize_word_vectors,
                   word_vector_save_path=args.word_vec_save_path,
                   max_loaded_word_vectors=args.max_word_vecs,
                   include_word_vectors=not args.omit_word_vectors,
                   include_chars=not args.omit_chars,
                   include_syntactical_features=not args.omit_syntactical_features,
                   include_exact_match=not args.omit_exact_match)
    else:
        raise ValueError('couldn\'t find implementation for specified dataset')
