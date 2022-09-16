# author:fighting
import numpy as np

def is_zh(word):
    if '\u4e00' <= word <= '\u9fa5':
            return True
    return False

# 读取词表
words_table = open('./data/public_simple_data/words_table', encoding='utf-8').read().split('\n')
# 设置一句话的长度为50
def get_word_weight(texts):
    # 计算其它字对目标字确定的影响
    max_length = 50
    result = []
    for sen in texts:  #句
        print('sen:', sen)
        word_arr = list(sen)
        temp1 = []
        n = 0;
        # for word1 in word_arr: #字
        for i in range(len(word_arr)):
            if i < 50:
                word1 = word_arr[i]
                print('word1:', word1)
                temp2 =[]
                for j in range(50): #其它字与该字的权重矩阵
                    if j < len(word_arr):
                        word2 = word_arr[j]
                        d = abs(word_arr.index(word1) - word_arr.index(word2)) + 1
                        if d > 1:
                            f = word_arr.index(word1) - word_arr.index(word2)
                            if f > 0:
                                str = ''.join(word_arr[word_arr.index(word2):word_arr.index(word1)+1])
                            else:
                                str = ''.join(word_arr[word_arr.index(word1):word_arr.index(word2)+1])
                            if is_in_words(str, words_table):
                                v = 1 + round(1/d, 6)
                                # print("sen:", sen, "words:", str)
                            elif d >= 2 and notInSenRight(str) and is_zh(word2):
                                print('str:', str)
                                v = - round(1/d, 6)
                            else:
                                v =  round(1/d, 6)
                        else: # 本身对本身决策的影响度设置为1
                            v = 0
                        temp2.append(v)
                    else:     # 句的长度小于50的时候填充1，当位置处于目标字的时候也填充1
                        k = 0
                        temp2.extend([k] * (max_length - len(word_arr)))
                        break
                # k = round(1/128, 4)
                # temp2 = normalization(temp2)
                print('temp2:', temp2)
                temp2 = np.asarray(temp2)
                temp1.append(temp2)
                n += 1
            else:  # 句的长度大于50不考虑
                break;
        if n < 50:
            temp3 = []
            for j in range(50-n):
                v1 = round(1/50, 6)
                # v1 = 1
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

def notInSenRight(str):
    for sen in sentences_right:
        if str in sen:
            return False
    return True

def is_in_words(str, wordss):
    for word in wordss:
        if str in word:
            return True
    return False;

# 获取正确语料
sentences13_right = open('./data/public_simple_data/rightSen13', encoding='utf-8').read().split('\n')
sentences15_right = open('./data/public_simple_data/rightSen15', encoding='utf-8').read().split('\n')
sentencesTest_right = open('./data/public_simple_data/rightTest', encoding='utf-8').read().split('\n')
yanzhengTest_right = open('./data/public_simple_data/test_yanzheng_right', encoding='utf-8').read().split('\n')
sentences_right = sentences13_right + sentences15_right + sentencesTest_right + yanzhengTest_right

# 读取不带标签的test文本
test_text_nolab = open('./data/public_simple_data/sentence_text', encoding='utf-8').read().split('\n')

tests = ['让我在学校的上课期间不会感到枯燥泛味', '不要白白让费这两天的假期']
test_weight = get_word_weight(tests)
# test_weight = get_word_weight(test_text_nolab)
print(test_weight[0][0])
print('test_weight.shape', test_weight.shape)