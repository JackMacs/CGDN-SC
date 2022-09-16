# author:fighting
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

def simlified2tradition(sentence):
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

def get_text(path):
    text = open(path).read().split('\n')
    return text

def change_fantilab_to_simplelab(read_path):
    sentences = open(read_path).read().split('\n')
    sens = []
    labs = []
    for sentence in sentences:
        sen = ''
        lab = []
        sentence = sentence.strip()
        temp = sentence.split(' ')
        for str_t in temp:
            str = str_t.split('/')
            # print('str_t:', str_t)
            sen = sen + str[0]
            lab.append(str[1])
        sen = simlified2tradition(sen)   #将lab文件中的数据进行简繁体转化
        sens.append(sen)
        labs.append(lab)
    return sens, labs

def save_simple_tofile(write_path, read_path):
    sens, labs = change_fantilab_to_simplelab(read_path)
    print('sens:', sens, 'labs:', labs)
    simple_sens = []
    j = 0
    for sen in sens:
        sen_arr = list(sen)
        lab = labs[j]
        str = ''
        for i in range(len(sen_arr)):
            if i < len(sen_arr)-1:
                str = str + sen_arr[i]+ '/' + lab[i] + ' '
            else:
                str = str + sen_arr[i] + '/' + lab[i]
        simple_sens.append(str)
        j += 1
    with open(write_path, 'w') as w:
        for simple_sen in simple_sens:
            w.write(simple_sen)
            w.write('\n')

# path = "/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/similarPronunciation.txt"
# write_text_path = '/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/simplePro.txt'
path = "/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/similarShape.txt"
write_text_path = '/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/simpleShape.txt'


confusion_texts = get_text(path)
train_texts = Simplified2Traditional(confusion_texts)
with open(write_text_path, 'w') as w:
    for sen in train_texts:
        w.write(sen)
        w.write('\n')

print('train_text:', train_texts)

# write_path = '../data/public_simple_data/test1_lab'
# read_path = '../data/public_data/public_test1_lab'
# save_simple_tofile(write_path, read_path)
