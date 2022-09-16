# author:fighting
import re
from pypinyin import lazy_pinyin
import os

def is_chinese(word):
    if '\u4e00' <= word <= '\u9fff':
        return True
    return False

# 获取汉字的拼音
def get_pinyin(word):
    result = lazy_pinyin(word)
    return result[0]


# 读取data_text数据并去重
# texts = open('./data/data_text').read().split('\n')
# texts = set(texts)
# print(len(texts))
# test = open('./data/rightTest').read().split('\n')
# sen13 = open('./data/rightSen13').read().split('\n')
# sen15 = open('./data/rightSen15').read().split('\n')
# texts = test + sen13 + sen15

# texts = open('../data/yt_data/data_text_noRepeat').read().split('\n')
tex1 = open('/Users/fighting/PycharmProjects/Text_error_detection/data/public_simple_data/rightSen13').read().split('\n')
tex2 = open('/Users/fighting/PycharmProjects/Text_error_detection/data/public_simple_data/rightSen15').read().split('\n')
tex3 = open('/Users/fighting/PycharmProjects/Text_error_detection/data/public_simple_data/rightDev').read().split('\n')
texts = tex1 + tex2 + tex3
dirname = '../word_vector/data/LCQMC'
sentence_LCQMC = []
words = []
for filename in os.listdir(dirname):
    with open(os.path.join(dirname, filename), 'r') as lcqmc:
        for line in lcqmc:
            linedict = eval(line)
            word = linedict['sentence1']
            poss = linedict['sentence2']
            sentence_LCQMC.append(word)
            sentence_LCQMC.append(poss)
# sentence_LCQMC = set(sentence_LCQMC)
# texts = list(texts)
# texts = texts + sentence_LCQMC


sentences = []
temp = []
for sen in texts:
    # 将句子以下列符号分割
    temp = re.split(',|，|？|!|！|。|【|】：', sen)
    for str in temp:
        if len(str.split())!=0:
            if is_chinese(str):
                sentences.append(str)
sentences = set(sentences)
print("noRepeat_size:", len(sentences))
print("sentences:", sentences)


# 将样本数据按照拼音，词形式保存形成同音词表
# 用于查找同音词
tongyin = {}

for sentence in sentences:
    sen_list = list(sentence)
    for word in sen_list:
        word_pinyin = get_pinyin(word)
        if word not in tongyin.keys():
            if is_chinese(word):
                tongyin[word] = word_pinyin


# # 将处理后的数据存入data_Norepeat文件中
# with open('./data/data_Norepeat', 'w') as w:
#     for str in sentences:
#         w.write(str)
#         w.write("\n")

string2_arr = []
temp = []
# 生成2元字符串存放在string2_arr并去重
for str in sentences:
    temp = list(str)
    length = len(temp)-1
    for i in range(length):
        s = temp[i]+ temp[i+1]
        string2_arr.append(s)
print(string2_arr)

string3_arr = []
temp = []
# 生成3元字符串放在string3_arr并去重
for str in sentences:
    temp = list(str)
    length = len(temp) - 2
    for i in range(length):
        s = temp[i] + temp[i+1] + temp[i+2]
        string3_arr.append(s)
print(string3_arr )

# 将2元串写入data_2string文件中
with open('./data/data_2string', 'w') as f:
    for str in string2_arr:
        f.write(str)
        f.write('\n')


# 将3元串写入data_3string文件中
with open('./data/data_3string', 'w') as f:
    for str in string3_arr:
        f.write(str)
        f.write('\n')

# 将tongyin字典中的词对应拼音存放于文件words_tab中
with open('./data/words_tab', 'w') as f:
    for key in tongyin:
        f.write(key+":"+tongyin[key])
        f.write('\n')


