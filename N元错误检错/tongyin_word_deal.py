# author:fighting
import re
from pypinyin import lazy_pinyin

def is_chinese(word):
    if '\u4e00' <= word <= '\u9fff':
        return True
    return False

# 获取汉字的拼音
def get_pinyin(word):
    result = lazy_pinyin(word)
    return result[0]


# test = open('./data/rightTest').read().split('\n')
# sen13 = open('./data/rightSen13').read().split('\n')
# sen15 = open('./data/rightSen15').read().split('\n')
# yanzheng = open('./data/test_yanzheng_right').read().split('\n')
sentences = []
temp = []
# texts = test + sen13 + sen15 + yanzheng
texts = open('../data/yt_data/data_text_noRepeat').read().split('\n')
for sen in texts:
    # 将句子以下列符号分割
    temp = re.split(',|，|？|!|！|。|：', sen)
    for str in temp:
        if len(str.split())!=0:
            sentences.append(str)
sentences = set(sentences)
print("样本sentences:", sentences)
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


# 将tongyin字典中的词对应拼音存放于文件words_tab中
with open('./data/words_tab', 'w') as f:
    for key in tongyin:
        f.write(key+":"+tongyin[key])
        f.write('\n')