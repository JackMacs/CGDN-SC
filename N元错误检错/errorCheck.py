# author:fighting
import re
import numpy as np
test_text = open('/Users/fighting/PycharmProjects/Text_error_detection/data/public_simple_data/test1_lab', encoding='utf-8').read().split('\n')
sen_nolab = []
sentences = []
labs = []
for strr in test_text:
    if len(strr.strip()) != 0:
        sentences.append(strr)

for sentence in sentences:
    groups = re.findall('(.)/(.)', sentence)
    arrays = np.asarray(groups)
    temp = ''.join(arrays[:, 0])  # 将数组转化为字符串
    temp1 = ''.join(arrays[:, 1])
    sen_nolab.append(temp)
    labs.append(temp1)
print(sen_nolab)
print("文本标签为：", labs)

# 读取2元和3元字符串
data_2str = open('data/data_2string', encoding='utf-8').read().split('\n')
data_3str = open('data/data_3string', encoding='utf-8').read().split('\n')

# 读取词拼音表
data_pinyins = open('data/words_tab', encoding='utf-8').read().split('\n')
# 读取形近字表
data_similarWords = open('data/similarWords_deal', encoding='utf-8').read().split('\n')

# 生成五笔编码列表
def get_encodeList(data_similarWords):
    words_encode = {}
    for strr in data_similarWords:
        temp = re.split(r':',strr)
        words_encode[temp[0]] = temp[1]
    return words_encode

# 生成同音词列表
def get_tongyinList(data_pinyins):
    words_pinyin = {}
    for strr in data_pinyins:
        if len(strr.strip()) != 0:
            temp = re.split(r':', strr)
            words_pinyin[temp[0]] = temp[1]
    return words_pinyin


def get_senErrorLoc(sentences):
    # 记录测试集中每个sentence中的可能存在的错误位置
    sen_error = {}
    f = 0
    for sentence in sen_nolab:
        error_pos = []
        for i in range(len(sentence)):
            index = len(sentence) -1
            if i > 0 and i < index:
                befor_2str = sentence[i-1 : i+1]
                # print("before:"+befor_2str)
                after_2str = sentence[i : i+2]
                # print("after:" + after_2str)
                if befor_2str in data_2str:
                    if after_2str in data_2str:
                        continue
                    else:
                        pos = i + 1
                        error_pos.append(pos)
                else:
                    pos = i + 1
                    error_pos.append(pos)
        sentence = sentence + str(f)
        f += 1
        sen_error[sentence] = error_pos
    return sen_error

# 计算两个字符串之间的编辑距离
def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(str1)][len(str2)]

# 获得某个字的同音字
def get_word_tongyin(word, words_pinyin):
    word_ty = []
    try:
        word_pin = words_pinyin[word]
    except:
        return word_ty
    for k,v in words_pinyin.items():
        if v == word_pin:
            word_ty.append(k)
    return word_ty

# 获得某个字的形近字
def get_word_similar(word, words_encode):
    word_sm = []
    try:
        word_en = words_encode[word]
    except:
        return word_sm
    for key in words_encode:
        if edit(word_en, words_encode[key]) == 1:
            word_sm.append(key)
    return word_sm

# 获取某个字的混淆集
def get_confuse_words(word, words_pinyin, words_encode):
    words_all = []
    word_ty = get_word_tongyin(word, words_pinyin)
    word_sm = get_word_similar(word, words_encode)
    words_all = word_ty + word_sm
    return words_all

# 获取字符串中的数字
def get_number(mystr):
    number = re.findall('\d+', mystr)[0]
    number = int(str(number))
    return number


# 求字符串中某个字串的个数
# str1 = "abskfirgnlskgabndf"
# str2 = "ab"
def getSubstr_num(str1, str2):
    num = (len(str1) - len(str1.replace(str2,"")))
    return num
# 遍历sen_error,存放训练样本和每个样本中可能存在的错误位置
sen_error = get_senErrorLoc(sen_nolab)
print('sen_nolab:',len(sen_nolab))

# 所有字的拼音字典
words_encode = get_encodeList(data_similarWords)
# 生成所有字的五笔编码编码字典
words_pinyin = get_tongyinList(data_pinyins)
print('五笔编码字典words_encode:', words_encode)
print('拼音字典words_pinyin:', words_pinyin)
tp = 0
fp = 0
fn = 0
tn = 0
right_sen = []
sens_correct = []
wrong_sen = []
print('可能存在的错误的位置：', sen_error)
f = 0  # 标记是否存在错误，0为不存在错误位置，1存在错误位置
real_wrong = {}
real_loc = []
i = -1  #记录当前处理sentence的文本
loc_num_p = 0; #记录每个sentence预测错误的个数
loc_num = 0; #记录每个sentence错误的个数

# sen_errs = {'转国内人公客服': [5, 6]}
predict_right = {}
predict_wrong = {}
output_str = []
h = 0 # 系统检测正确的文本数目
lce = 0 # 系统正确定位到错误字符的个数
num = 0
for key in sen_error:
# for key in sen_errs:
    i += 1
    # print('i:', i)
    location = sen_error[key]
    output_str = []
    loc_num_p = 0
    if len(location) == 0:
        # if 'e' not in labs[i]:
        #     h += 1
        # if 'e' in labs[i]:
        #     fn += 1
        # else:
        #     tn += 1
        #     right_sen.append(key)
        # print('识别为正确：',key, labs[i])
        predict_right[key] = labs[i]
    else:
        for loc in location:
            key_arr = list(key)
            index = loc-1
            word = key_arr[index]
            candidate_word = get_confuse_words(word, words_pinyin, words_encode)
            # print('candidate word:', candidate_word)

            for canStr in candidate_word:
                temp = key_arr[index-1]+canStr+key_arr[index+1]
                # print('temp:', temp)
                if temp in data_3str:
                    # f = 1
                    loc_num_p += 1
                    # if key in predict_right.keys():
                    output_str.append(labs[i])
                    predict_wrong[key] = labs[i]
                    num += 1
                    print(key+":"+temp+":"+str(num))
                    break
                    # print('canStr:', canStr)
                    # key_arr[index] = canStr
                    # print("key_arr", key_arr)
                    # real_loc.append(index)
        if len(output_str) == 0:
            predict_right[key] = labs[i]
    # real_wrong[key] = real_loc
    # print("key_arr2", key_arr)
    loc_num = getSubstr_num(labs[i], 'e')
    if loc_num_p == loc_num:
        if loc_num_p != 0:
            lce += 1
        h += 1
print('sen_error_size:', len(sen_error))
print('系统正确检测的sentence数目：h', h)
print('系统正确定位错误字符的数目：', lce)
print("predict_right:", predict_right)
print("predict_wrong:", predict_wrong)
TE = 0
SE = 0  # of sentences the evaluated system reported to have errors
DC = 0  # of sentences with correctly detected results
DE = 0  # of sentences with correctly detected errors
FPE = 0  # of sentences with false positive errors
ANE = 0  # of testing sentences without errors
ALL = 0  # of all testing sentences
AWE = 0  # of testing sentences with errors
CLD = 0  # of sentences with correct location detection
CEL = 0  # of sentences with correct error locations
ALL = len(labs)
y_predict_text = {}
# y_predict_text = [0] * ALL
# y_predict_text = predict_wrong + predict_right
for key in predict_wrong:
    y_predict_text[key] = predict_wrong[key]
for key in predict_right:
    y_predict_text[key] = predict_right[key]
# for key in predict_wrong:
#     index = get_number(key)
#     y_predict_text[index] = predict_wrong[key]
# for key in predict_right:
#     index = get_number(key)
#     y_predict_text[index] = predict_right[key]
print('labs:', labs)
print('y_predict_text:', y_predict_text)
print('y_predict_text_size:', len(y_predict_text))
print('predict_right_size:', len(predict_right))
print('predict_wrong_size:', len(predict_wrong))

i = 0
SE = len(predict_wrong)
for er in labs:
    if 'e' in er:
        TE += 1
    if 'e' not in er:
        ANE += 1
for key in predict_wrong:
    if 'e' in predict_wrong[key]:
        DC += 1
for key in predict_right:
    if 'e' not in predict_right[key]:
        DC += 1
for key in predict_wrong:
    if 'e' in predict_wrong[key]:
        DE += 1
# for key in predict_right:
#     if 'e' in predict_right[key]:
#         FPE += 1
for key in predict_wrong:
    if 'e' not in predict_wrong[key]:
        FPE += 1
AWE = TE
CLD = h
CEL = lce

print('SE:', SE, 'TE:', TE, 'DE:', DE, 'DC:', DC, 'FPE:', FPE, 'ANE:', ANE,'ALL:', ALL,  'AWE:', AWE, 'CLD:', CLD, 'CEL:', CEL)
FAR = FPE / ANE
DA  =  DC / ALL
DP = DE / SE
DR = DE / TE
DF1 = 2 * DP * DR / (DP + DR)
ELA = CLD / ALL
ELP = CEL / SE
ELR = CEL / TE
ELF1 = 2 * ELP * ELR / (ELP + ELR)
print('FAR:', FAR, 'DA:', DA, 'DP:', DP, 'DR:', DR, 'DF1:', DF1, 'ELA:', ELA, 'ELP:', ELP, 'ELR:', ELR, 'ELF1:', ELF1)
for key in predict_right:
    if 'e' in predict_right[key]:
        fn += 1
    if 'e' not in predict_right[key]:
        tn += 1
for key in predict_wrong:
    if 'e' in predict_wrong[key]:
        tp += 1
    if 'e' not in predict_wrong[key]:
        fp += 1

count = len(predict_wrong) +len(predict_right)
print("预测总量：", count)
# print("right_sen:", right_sen)
# # print("sens_correct:", sens_correct)
# print('wrong_sen:', wrong_sen)
# print('准确率：', (len(wrong_sen)+len(right_sen))/120)
print('TP:', tp,'FP:', fp, 'FN:', fn, 'TN:',tn)
fpr = fp / (fp + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * precision * recall) / (precision + recall)
acc = h /len(sen_error)
print('FPR的值为：', fpr)
print('Precision:', precision)
print('Accuracy的值：', acc)
print('recall的值：', recall)
print('f1的值：', f1)
