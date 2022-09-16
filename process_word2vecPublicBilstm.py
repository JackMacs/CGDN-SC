import re
from itertools import chain
import pandas as pd
import numpy as np
import pickle
import random
from collections import Counter
from gensim.models.word2vec import Word2Vec


# Read origin data13
# 将带有标签的样本数据转化为列表
sentences, sen_test = [], []

# 读取data_tag作为训练数据
text = open('./data/public_simple_data/sentence_lab2', encoding='utf-8').read().split('\n')  # 消除换行符

# 读取训练文本数据
train_text = open('./data/public_simple_data/sentence_text2', encoding='utf-8').read().split('\n')

# 读取data_test作为测试数据
# test_text = open('./data/public_simple_data/test1_lab', encoding='utf-8').read().split('\n')   #单独使用测试文本
test_text = open('correctionData/test_text2', encoding='utf-8').read().split('\n')

# 读取data_dev作为验证数据
dev_test = open('./data/public_simple_data/dev_lab', encoding='utf-8').read().split('\n')
# 读取不带标签的验证数据
dev_text_nolab = open('./data/public_simple_data/dev_text', encoding='utf-8').read().split('\n')

# 读取不带标签的test文本
test_text_nolab = open('./data/public_simple_data/test1_text', encoding='utf-8').read().split('\n')
test_text_ = open('correctionData/test_text2', encoding='utf-8').read().split('\n')
# 读取词表
words_table = open('./data/public_simple_data/words_table', encoding='utf-8').read().split('\n')

# 获取正确语料
sentences13_right = open('./data/public_simple_data/rightSen13', encoding='utf-8').read().split('\n')
sentences15_right = open('./data/public_simple_data/rightSen15', encoding='utf-8').read().split('\n')
sentencesTest_right = open('./data/public_simple_data/rightTest', encoding='utf-8').read().split('\n')
sentencesDev_right = open('./data/public_simple_data/rightDev', encoding='utf-8').read().split('\n')
sentencesYanzheng_right = open('./data/public_simple_data/test_yanzheng_right', encoding='utf-8').read().split('\n')
sentences_right = sentences13_right + sentences15_right + sentencesTest_right + sentencesDev_right + sentencesYanzheng_right

def notInSenRight(str):
    for sen in sentences_right:
        if str in sen:
            return False
    return True

train_sentence_length, dev_sentence_length, test_sentence_length = [], [], []

# 获取不带标签的训练样本的长度
for str in train_text:
    if len(str.strip()) != 0:
        train_sentence_length.append(len(str))
print('train_sen_len:', len(train_sentence_length))


# sentences中存放带标签的训练数据
f = 1
for str in text:
    if len(str.strip()) != 0:
        sentences.append(str)


for str in test_text:
    if len(str.strip()) != 0:
        sen_test.append(str)


# 获取无标签验证样本的长度列表
for str in dev_text_nolab:
    if len(str.strip()) != 0:
        dev_sentence_length.append(len(str))
print('dev_sen_len:', len(dev_sentence_length))

for str in test_text:
    if len(str.strip()) != 0:
        test_sentence_length.append(len(str))
print('test_sen_len:', len(test_sentence_length))

print('train_sentence_size:', len(train_sentence_length))
print('train_sentence_length:', train_sentence_length)
print('test_sentence_length:', test_sentence_length)
print('dev_sentence_length:', dev_sentence_length)


# To numpy array
words, labels, words_test, words_dev, dev_label= [], [], [], [], []
print('Start creating words and labels...')
print("sentence_size:", len(sentences))
for sentence in sentences:
    groups = re.findall('(.)/(.)', sentence)
    arrays = np.asarray(groups)
    words.append(arrays[:, 0])
    labels.append(arrays[:, 1])
print('Words Length', len(words), 'Labels Length', len(labels))
print('Words Example', words[0])
print('Labels Example', labels[0])

# for sentence in sen_test:
#     groups = re.findall('(.)/(.)', sentence)
#     arrays = np.asarray(groups)
#     words_test.append(arrays[:, 0])
#     test_label.append(arrays[:, 1])
for str in test_text:
    temp = list(str)
    words_test.append(np.asarray(temp))


for sen in dev_test:
    groups = re.findall('(.)/(.)', sentence)
    arrays = np.asarray(groups)
    words_dev.append(arrays[:, 0])
    dev_label.append(arrays[:, 1])

all_words = words + words_test + words_dev
td_words = words + words_dev
# Merge all words
all_words = list(chain(*all_words))
# All words to Series
all_words_sr = pd.Series(all_words)
# Get value count, index changed to set
all_words_counts = all_words_sr.value_counts()
# print("all_words_counts:",all_words_counts)
# Get words set
all_words_set = all_words_counts.index
# Get words ids
all_words_ids = range(1, len(all_words_set) + 1)

# Dict to transform (word, id)
word2id = pd.Series(all_words_ids, index=all_words_set)

#(id, word)
id2word = pd.Series(all_words_set, index=all_words_ids)

# Tag set and ids
# tags_set = ['x', 's', 'b', 'm', 'e']
tags_set = ['x', 'e', 'r']
tags_ids = range(len(tags_set))

# Dict to transform
# (tag, id)
tag2id = pd.Series(tags_ids, index=tags_set)

# (id, tag)
id2tag = pd.Series(tags_set, index=tag2id)

# 处理一句话的长度为50字符
max_length = 50


def is_zh(word):
    if '\u4e00' <= word <= '\u9fa5':
            return True
    return False

model = Word2Vec.load('word_vector/wordVec_model/word2vecModel_pub')
def wordToVector(words):
    # print('words:',words)
    result = []
    for senarr in words:
        temp = []
        for i in range(50):
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



def x_transform(wordss):
    ids = list(word2id[wordss])
    if len(ids) >= max_length:
        ids = ids[:max_length]
    ids.extend([0] * (max_length - len(ids)))
    return ids


def y_transform(tags):
    ids = list(tag2id[tags])
    if len(ids) >= max_length:
        ids = ids[:max_length]
    ids.extend([0] * (max_length - len(ids)))
    return ids

# 将标签转化为one-hot编码
def transform_one_hot(labels):
    one_hot = np.eye(3)[labels]
    return one_hot

# 字权重归一化
def normalization(list):
    result = []
    t = 0
    for a in list:
        t = t+a
    for a in list:
        if t == 0:
            temp = 0
        else:
            temp = round(a/t,4)
        result.append(temp)
    return result

def is_in_words(str, wordss):
    for word in wordss:
        if str in word:
            return True
    return False

def word_isIN_corpus(word, words):
    for w in words:
        if word in w:
            return True
    return False


# 设置一句话的长度为50
def get_word_weight(texts):
    # 计算其它字对目标字确定的影响
    max_length = 50
    result = []
    for sen in texts:  #句
        print('sen:', sen)
        word_arr = list(sen)
        temp1 = []
        n = 0
        # for word1 in word_arr: #字
        for i in range(len(word_arr)):
            if i < 50:
                word1 = word_arr[i]
                temp2 =[]
                for j in range(50): #其它字与该字的权重矩阵
                    if j < len(word_arr):
                        word2 = word_arr[j]
                        # d = abs(word_arr.index(word1) - word_arr.index(word2)) + 1
                        d = abs(i - j) + 1
                        if d > 1:
                            # f = word_arr.index(word1) - word_arr.index(word2)
                            f = i - j
                            if f > 0:
                                # str = ''.join(word_arr[word_arr.index(word2):word_arr.index(word1)+1])
                                str = ''.join(word_arr[j:i+1])
                            else:
                                # str = ''.join(word_arr[word_arr.index(word1):word_arr.index(word2)+1])
                                str = ''.join(word_arr[i:j + 1])
                            if is_in_words(str, words_table):
                                # v = d + round(1/d, 6)
                                v = 1 + round(1/d, 6)
                                # print("sen:", sen, "words:", str)
                            elif d >= 2 and notInSenRight(str) and is_zh(word2):
                                v = - round(1/d, 6)
                            else:
                                # v = 1 + round(1/d, 6)
                                v = round(1/d, 6)
                                # v = 1
                                # v = 0
                        else: # 本身对本身决策的影响度设置为1
                            # v = 1
                            v = 0
                        temp2.append(v)
                    else:     # 句的长度小于50的时候填充1，当位置处于目标字的时候也填充1
                        k = 0
                        temp2.extend([k] * (max_length - len(word_arr)))
                        break
                # k = round(1/128, 4)
                # temp2 = normalization(temp2)
                temp2 = np.asarray(temp2)
                temp1.append(temp2)
                n += 1
            else:  # 句的长度大于50不考虑
                break;
        if n < 50:
            temp3 = []
            for j in range(50-n):
                # v1 = round(1/50, 6)
                v1 = 0
                temp3 = [v1 for _ in range(max_length)]
                temp3 = np.asarray(temp3)
                temp1.append(temp3)
        if len(temp1)!= 0:
            temp1 = np.asarray(temp1)
            # print('temp1.shape:', len(temp1))
        # 实现temp2的转置
        # temp1 = temp1.transpose()
        # print(sen)
        tm = np.asarray(temp1)
        # print('tm.shape:', tm.shape)
        result.append(tm)
    result = np.asarray(result)
    return result

# def get_word_weight(texts):
#     # 计算其它字对目标字确定的影响
#     max_length = 128
#     result = []
#     for sen in texts:  #句
#         word_arr = list(sen)
#         temp1 = []
#         n = 0;
#         # for word1 in word_arr: #字
#         for i in range(len(word_arr)):
#             if i < 50:
#                 word1 = word_arr[i]
#                 temp2 =[]
#                 for j in range(128): #其它字与改字的权重矩阵
#                     if j < len(word_arr):
#                         word2 = word_arr[j]
#                     else:
#                         break
#                     d = abs(word_arr.index(word1) - word_arr.index(word2))+1
#                     if d > 1:
#                         f = word_arr.index(word1) - word_arr.index(word2)
#                         if f > 0:
#                             str = ''.join(word_arr[word_arr.index(word2):word_arr.index(word1)+1])
#                         else:
#                             str = ''.join(word_arr[word_arr.index(word1):word_arr.index(word2)+1])
#                         if is_in_words(str, words_table):
#                             v = d + round(1/d,6)
#                             # print("sen:", sen, "words:", str)
#                         else:
#                             v = 1 + round(1/d, 6)
#                     else:
#                         v = 1
#                     temp2.append(v)
#                 # k = round(1/128, 4)
#                 k = 1
#                 temp2.extend([k] * (max_length - len(word_arr)))
#                 temp2 = normalization(temp2)
#                 temp2 = np.asarray(temp2)
#                 temp1.append(temp2)
#                 n += 1
#             else:
#                 break;
#         if n < 50:
#             for j in range(50-n):
#                 v1 = round(1/128, 6)
#                 # v1 = 1
#                 temp2 = [v1 for _ in range(max_length)]
#                 temp2 = np.asarray(temp2)
#                 temp1.append(temp2)
#         temp1 = np.asarray(temp1)
#         # 实现temp2的转置
#         # temp1 = temp1.transpose()
#         print(sen)
#         result.append(temp1)
#     result = np.asarray(result)
#     return result

print('Starting transform...')
# words为sentences集合
# data_x = list(map(lambda x: x_transform(x), words))
data_x = wordToVector(words)
# labels为标签集合
data_y = list(map(lambda y: y_transform(y), labels))

dev_x = wordToVector(words_dev)
test_x = wordToVector(words_test)
# test_y = list(map(lambda y: y_transform(y), test_label))
dev_y = list(map(lambda y: y_transform(y), dev_label))

print('Data Y Length', len(data_y))


print('Data Y Example', data_y[0])
dev_x = np.asarray(dev_x)
test_x = np.asarray(test_x)
data_x = np.asarray(data_x)
print('dev_x_shape:', dev_x.shape)
print('test_x_shape:', test_x.shape)
print('data_x_shape', data_x.shape)
# data_y = transform_one_hot(data_y)
data_y = np.asarray(data_y)
# test_y = np.asarray(test_y)
dev_y = np.asarray(dev_y)
print("data_y", data_y[0])
print('data_y_shape', data_y.shape)
# print('test_y_shape:',test_y.shape)
print('dev_y_shape:',dev_y.shape)
print('dev_y_len:', len(dev_y))

from os import makedirs
from os.path import exists, join

path = 'data/'

if not exists(path):
    makedirs(path)

# word_weight = get_word_weight(texts)
# print("word_weight_shape:", word_weight.shape)
print("texts:", test_text_nolab)
print("texts_size:", len(test_text_nolab))

train_weight = get_word_weight(train_text)
test_weight = get_word_weight(test_text_nolab)
print('test_weight.shape', test_weight.shape)
print('train_weight.shape', train_weight.shape)


dev_weight = get_word_weight(dev_text_nolab)
print('dev_weight.shape', dev_weight.shape)



print('test_weight.shape', test_weight.shape)
# print('train_weight.shape', train_weight.shape)
print("test_weight[0][1]:", test_weight[0][1])
print('Starting pickle to file...')
with open(join(path, 'data_word3vec.pkl'), 'wb') as f:
    pickle.dump(data_x, f)
    pickle.dump(data_y, f)
    pickle.dump(test_x, f)
    # pickle.dump(test_y, f)
    pickle.dump(dev_x, f)
    pickle.dump(dev_y, f)
    # pickle.dump(test_text_nolab, f)
    pickle.dump(test_text_, f)
    pickle.dump(word2id, f)
    pickle.dump(tag2id, f)
    pickle.dump(id2tag, f)
    pickle.dump(test_weight, f)
    pickle.dump(train_weight, f)
    pickle.dump(dev_weight, f)
    pickle.dump(train_sentence_length, f)
    pickle.dump(test_sentence_length, f)
    pickle.dump(dev_sentence_length, f)
print('Pickle finished')
