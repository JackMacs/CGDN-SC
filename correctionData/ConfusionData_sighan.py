simP = open('/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/simplePro.txt',
                      encoding='utf-8').read().split('\n')
# confusionSet = open('/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/confusion1.txt', encoding='utf-8').read().split('\n')
# # write_text_path = '/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/simpleProDeal.txt'
write_text_path = '/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/confusion3.txt'
simShape = open('/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/simpleShape.txt', encoding='utf-8').read().split('\n')

def listTodict(cs, deli):
    ls = []
    for str in cs:
        print(str)
        sarr = str.split(deli)
        temp = []
        temp.append(sarr[0])
        t = ""
        for i in range(1, len(sarr)):
            t = t + sarr[i]
            # temp.append(sarr[1])
        temp.append(t)
        ls.append(temp)
    return dict(ls)
simProM = listTodict(simP, "\t")
simShape = listTodict(simShape, ",")
# print("simShape:", simShape)
for key in simShape:
    # print("hu")
    # print(key + ":" + simShape.get(key))
    if key in simProM.keys():
        value = simShape.get(key)
        # print("value:" + value)
        varr = list(value)
        cv = simProM.get(key)
        for v in varr:
            if v not in cv:
                # print("key1:"+cv)
                cv = cv+v
                simProM[key] = cv
    else:
        # print("不在困惑集中的数据")
        # print("value:", simShape.get(key))
        # print(key+":"+simShape.get(key))
        simProM[key] = simShape.get(key)
# print("simProM:", simProM)
for key in simProM:
    print(key + ":" + simProM.get(key))
with open(write_text_path, 'w') as w:
    for key in simProM:
        # print("hu")
        # print(simProM.get(key))
        w.write(key+":"+simProM.get(key))
        w.write("\n")
#
#         w.write('\n')