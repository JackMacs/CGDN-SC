# author:fighting
import jieba
from langconv import *
def Simplified2Traditional(sentences):
    '''
    将sentence中的简体字转为繁体字
    :param sentence: 待转换的句子
    :return: 将句子中简体字转换为繁体字之后的句子
    '''
    sentence_arr = []
    for sentence in sentences:
        sentence = Converter('zh-hans').convert(sentence)
        sentence_arr.append(sentence)
    return sentence_arr

def save_to_file(path, sentences):
    with open(path, 'w') as w:
        for sen in sentences:
            w.write(sen)
            w.write('\n')

sentences_right13 = open('../data/public_data/public_rightSen13').read().split('\n')
sentences_right15 = open('../data/public_data/public_rightSen15').read().split('\n')
sentences_rightTest = open('../data/public_data/public_rightTest').read().split('\n')
sentences_right_test1 = open('../data/public_simple_data/test_yanzheng_right').read().split('\n')
sentences_right = sentences_right13 + sentences_right15 + sentences_rightTest + sentences_right_test1
sentences_right = Simplified2Traditional(sentences_right)  #将繁体的句子集合改为简体的句子集合
sentence_simple13 = Simplified2Traditional(sentences_right13)
sentence_simple15 = Simplified2Traditional(sentences_right15)
sentence_simpleTest = Simplified2Traditional(sentences_rightTest)

path13 = '../data/public_simple_data/rightSen13'
path15 = '../data/public_simple_data/rightSen15'
path_test = '../data/public_simple_data/rightTest'
save_to_file(path13, sentence_simple13)
save_to_file(path15, sentence_simple15)
save_to_file(path_test, sentence_simpleTest)

print('sentence_right:', sentences_right)
words = []
for sen in sentences_right:
    seg_list = jieba.cut(sen, cut_all=False)
    words.extend(seg_list)

# words = words + wordss
words = set(words)

with open('../data/public_simple_data/words_table', 'w') as w:
    for word in words:
        w.write(word)
        w.write('\n')