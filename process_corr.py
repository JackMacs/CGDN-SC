import re
from itertools import chain
from os.path import join

import pandas as pd
import numpy as np
import pickle
import random
from collections import Counter
from gensim.models.word2vec import Word2Vec
# Read origin data13
# 将带有标签的样本数据转化为列表
sentences, sen_test = [], []
# 读取训练文本数据
train_text = open('/Users/fighting/PycharmProjects/Text_error_detection/correctionData/train_data.txt', encoding='utf-8').read().split('\n')
# 将训练集中的数据打乱
random.shuffle(train_text)
# 将训练数据中的文本和标签分离
trains = []
labels = []
for str in train_text:
    sarr = str.split('@')
    text = sarr[0]
    print(sarr)
    label = int(sarr[1])
    tarr = list(text)
    tarr = np.asarray(tarr)
    trains.append(tarr)
    labels.append(label)

# To numpy array
# words, labels, words_test, test_label, words_dev, dev_label= [], [], [], [], [], []
# print('Start creating words and labels...')
# print("sentence_size:", len(sentences))
# for sentence in sentences:
#     groups = re.findall('(.)/(.)', sentence)
#     arrays = np.asarray(groups)
#     words.append(arrays[:, 0])
#     labels.append(arrays[:, 1])
# print('Words Length', len(words), 'Labels Length', len(labels))
# print('Words Example', words[0])
# print('Labels Example', labels[0])
#
# for sentence in sen_test:
#     groups = re.findall('(.)/(.)', sentence)
#     arrays = np.asarray(groups)
#     words_test.append(arrays[:, 0])
#     test_label.append(arrays[:, 1])
#
#
# for sen in dev_test:
#     groups = re.findall('(.)/(.)', sentence)
#     arrays = np.asarray(groups)
#     words_dev.append(arrays[:, 0])
#     dev_label.append(arrays[:, 1])
#
# all_words = words + words_test + words_dev
# td_words = words + words_dev
# # Merge all words
# all_words = list(chain(*all_words))
# # All words to Series
# all_words_sr = pd.Series(all_words)
# # Get value count, index changed to set
# all_words_counts = all_words_sr.value_counts()
# # print("all_words_counts:",all_words_counts)
# # Get words set
# all_words_set = all_words_counts.index
# # Get words ids
# all_words_ids = range(1, len(all_words_set) + 1)
#
# # Dict to transform (word, id)
# word2id = pd.Series(all_words_ids, index=all_words_set)
#
# #(id, word)
# id2word = pd.Series(all_words_set, index=all_words_ids)
#
# # Tag set and ids
# # tags_set = ['x', 's', 'b', 'm', 'e']
# tags_set = ['x', 'e', 'r']
# tags_ids = range(len(tags_set))
#
# # Dict to transform
# # (tag, id)
# tag2id = pd.Series(tags_ids, index=tags_set)
#
# # (id, tag)
# id2tag = pd.Series(tags_set, index=tag2id)
#
# 处理一句话的长度为50字符
max_length =30
#
#
# def is_zh(word):
#     if '\u4e00' <= word <= '\u9fa5':
#             return True
#     return False
#
#
model = Word2Vec.load('word_vector/wordVec_model/word2vecModel_yt')
def wordToVector(words):
    # print('words:',words)
    result = []
    for senarr in words:
        temp = []
        for i in range(30):
            if i < len(senarr):
                try:
                    word_vec = model[senarr[i]]
                    word_vec = np.asarray(word_vec)
                except:
                    print('word2vec no word:', senarr[i])
                    word_vec = np.random.random(128)
                    print('word_vec_shape:', word_vec.shape)
                    print('word_vec:', word_vec)
            else:
                word_vec = [0 for _ in range(128)]
            temp.append(word_vec)
        # temp = np.asarray(temp)
        result.append(temp)
    return result



# 将标签转化为one-hot编码
def transform_one_hot(labels):
    one_hot = np.eye(3)[labels]
    return one_hot
#
#
# print('Starting transform...')
data_x = wordToVector(trains)
data_y = labels
# # labels为标签集合
# data_y = list(map(lambda y: y_transform(y), labels))
#
# dev_x = wordToVector(words_dev)
# test_x = wordToVector(words_test)
# test_y = list(map(lambda y: y_transform(y), test_label))
# dev_y = list(map(lambda y: y_transform(y), dev_label))
#
#
# print('Data Y Example', data_y[0])
# dev_x = np.asarray(dev_x)
# test_x = np.asarray(test_x)
data_x = np.asarray(data_x)
# print('dev_x_shape:', dev_x.shape)
# print('test_x_shape:', test_x.shape)
print('data_x_shape', data_x.shape)
# # data_y = transform_one_hot(data_y)
data_y = np.asarray(data_y)
print('data_y_shape', data_y.shape)
# test_y = np.asarray(test_y)
# dev_y = np.asarray(dev_y)
# print("data_y", data_y[0])
# print('data_y_shape', data_y.shape)
# print('test_y_shape:',test_y.shape)
# print('dev_y_shape:',dev_y.shape)
# print('dev_y_len:', len(dev_y))
#
# from os import makedirs
# from os.path import exists, join
#
path = '/Users/fighting/PycharmProjects/Text_error_detection/data/'
#
# if not exists(path):
#     makedirs(path)
#
# # word_weight = get_word_weight(texts)
# # print("word_weight_shape:", word_weight.shape)
# print("texts:", test_text_nolab)
# print("texts_size:", len(test_text_nolab))
#
#
print('Starting pickle to file...')
with open(join(path, 'data_sighan_corr.pkl'), 'wb') as f:
    # max_bytes = 2 ** 31 - 1
    # data = bytearray(data_x)
    # bytes_out = pickle.dump(data, f)
    # for idx in range(0, len(bytes_out), max_bytes):
    #     f.write(bytes_out[idx:idx + max_bytes])
    pickle.dump(data_x, f)
    pickle.dump(data_y, f)
# print('Pickle finished')