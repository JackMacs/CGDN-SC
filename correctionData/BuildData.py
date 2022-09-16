import re
import numpy as np
# 读取训练文本数据
train_text = open('/Users/fighting/PycharmProjects/Text_error_detection/data/yt_data/data_train', encoding='utf-8').read().split('\n')
# 读取data_test作为测试数据
test_text = open('/Users/fighting/PycharmProjects/Text_error_detection/data/yt_data/data_test', encoding='utf-8').read().split('\n')   #单独使用测试文本
# 读取data_dev作为验证数据
dev_test = open('/Users/fighting/PycharmProjects/Text_error_detection/data/yt_data/data_dev', encoding='utf-8').read().split('\n')
confusion_txt = open('/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/confusion.txt', encoding='utf-8').read().split('\n')
data_text_noRepeat = open('/Users/fighting/PycharmProjects/Text_error_detection/data/yt_data/data_text_noRepeat', encoding='utf-8').read().split('\n')
def getConfusionMap(confusion_txt):
    ls = []
    for str in confusion_txt:
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

def getCandidates(text, tags):
    cs = []
    d = getConfusionMap(confusion_txt)
    for i in tags:
        if len(cs) == 0:
            error = text[i]
            if 1-(error in d.keys()):
                cs.append(text)
                return cs
            ls = d[error]
            print("error_conset")
            print(ls)
            flag = len(ls)
            if flag > 20:
                flag = 20
            # for c in ls:
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
                ls = d[error]
                flag = len(ls)
                if flag > 4:
                    flag = 4
                # for c in ls:
                for j in range(flag):
                    temp = str
                    temp = list(temp)
                    temp[i] = ls[j]
                    temp = ''.join(temp)
                    cs.append(temp)
    cs.append(text)
    return cs



train, test, dev = [], [], []
def getDataset(text):
    i = 0
    for s in text:
        if "e" in s:
            print(s)
            str1, tag1 = getText(s)
            cs = getCandidates(str1, tag1)
            for cstr in cs:
                train.append(cstr+"@0")
        else:
            str2, tag2 = getText(s)
            train.append(str2 + "@1")
    return train

def writeFile(texts):
    with open('train_data.txt', 'w') as wf:
        for str in texts:
            wf.write(str)
            wf.write('\n')

train = getDataset(train_text)
test = getDataset(test_text)
dev = getDataset(dev_test)

corrTexs = []
for str in data_text_noRepeat:
    corrTexs.append(str+"@1")
ts = set(corrTexs + train + test + dev)

tets = set(train + test + dev)
print(len(tets))
print(len(corrTexs))
print(len(ts))
# writeFile(train)
# writeFile(test)
# writeFile(dev)
# writeFile(corrTexs)
writeFile(ts)


# sentence = '我/r 想/r 差/e 异/e 下/r 快/r 递/r 物/r 流/r'
# tes, tag = getText(sentence)
# print(getCandidates(tes, tag))
# print(tag)


