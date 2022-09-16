import random
import re


import numpy as np
# 读取训练文本数据
train_text = open('/Users/fighting/PycharmProjects/Text_error_detection/data/public_simple_data/test1_lab', encoding='utf-8').read().split('\n')
# 读取data_test作为测试数据
# test_text = open('/Users/fighting/PycharmProjects/Text_error_detection/data/yt_data/data_test', encoding='utf-8').read().split('\n')   #单独使用测试文本
# 读取data_dev作为验证数据
dev_test = open('/Users/fighting/PycharmProjects/Text_error_detection/data/public_simple_data/dev_lab', encoding='utf-8').read().split('\n')
confusion_txt = open('/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/confusion3.txt', encoding='utf-8').read().split('\n')
btrain_text = open('/Users/fighting/PycharmProjects/Text_error_detection/correctionData/btrain_sighan14.txt', encoding='utf-8').read().split('\n')
ctrain_text = open('/Users/fighting/PycharmProjects/Text_error_detection/correctionData/ctrain_sighan14.txt', encoding='utf-8').read().split('\n')

def getConfusionMap(confusion_txt):
    ls = []
    for str in confusion_txt:
        # print(str)
        sarr = str.split(':')
        temp = []
        temp.append(sarr[0])
        temp.append(sarr[1])
        ls.append(temp)
    return dict(ls)

def getText(sentence):
    tas = []
    groups = re.findall('(.)/(.)', sentence)
    arrays = np.asarray(groups)
    arr = (arrays[:, 0])
    tagArr = (arrays[:, 1])
    str = ''.join(arr)
    tagStr = ''.join(tagArr)
    starts = [each.start() for each in re.finditer("e", tagStr)]
    return str, starts


def getCandidates_new(text):
    cs = []
    d = getConfusionMap(confusion_txt)
    for j in range(1):
        index = random.randint(0, len(text)-1)
        # print(text)
        # print(len(text))
        # print(index)
        zf = text[index]
        if zf in d.keys():
            ls = d.get(zf)
            flag = -1
            if len(ls) > 2:
                flag = 2
            for j in range(flag):
            # for ef in ls:
                temp = text
                temp = list(temp)
                temp[index] = ls[j]
                temp = ''.join(temp)
                if temp not in cs:
                    cs.append(temp)
    return cs

# tag中存放每个sentence中错误字符字符所处的位置
def getCandidates(text, tags):
    cs = []
    d = getConfusionMap(confusion_txt)
    for i in tags:
        if len(cs) == 0:
            error = text[i]
            if error in d.keys():
                cs.append(text)
                return cs
            ls = d[error]
            flag = len(ls)
            for j in range(flag):
                temp = text
                temp = list(temp)
                temp[i] = ls[j]
                temp = ''.join(temp)
                cs.append(temp)
        else:
            t_cs = cs.copy()
            for str in t_cs:
                error = text[i]
                if 1 - (error in d.keys()):
                    cs.append(text)
                    return cs
                ls = d[error]
                flag = len(ls)
                if flag > 0:
                    flag = 0
                # for c in ls:
                for j in range(flag):
                    temp = str
                    temp = list(temp)
                    temp[i] = ls[j]
                    temp = ''.join(temp)
                    cs.append(temp)
    cs.append(text)
    return cs

# def getDataset(text):
#     i = 0
#     for s in text:
#         if "e" in s:
#             # print(s)
#             str1, tag1 = getText(s)
#             cs = getCandidates(str1, tag1)
#             for cstr in cs:
#                 train.append(cstr+"@0")
#         else:
#             str2, tag2 = getText(s)
#             train.append(str2 + "@1")
#     return train

def getDataset(text):
    i = 0
    for s in text:
        if "e" in s:
            # print(s)
            str1, tag1 = getText(s)
            # cs = getCandidates_new(str1, tag1)
            # for cstr in cs:
            #     train.append(cstr+"@0")
            train.append(str1 + "@0")
        else:
            str2, tag2 = getText(s)
            train.append(str2 + "@1")
            cs = getCandidates_new(str2)
            for cstr in cs:
                train.append(cstr+"@0")
    return train

def writeFile(texts):
    with open('train_data_sighan.txt', 'w') as wf:
        for str in texts:
            wf.write(str)
            wf.write('\n')


train, dev = [], []
train = getDataset(train_text)
dev = getDataset(dev_test)
btrain = getDataset(btrain_text)
ctrain = getDataset(ctrain_text)
ts = set(train + dev + btrain + ctrain)
i = 0
for s in ts:
    if "1" in s:
        i += 1
print(len(ts))
print(i)
# writeFile(train)
# writeFile(test)
# writeFile(dev)
# writeFile(corrTexs)
writeFile(ts)



