simPro = open('/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/simplePro.txt',
                      encoding='utf-8').read().split('\n')
confusionSet = open('/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/confusion1.txt', encoding='utf-8').read().split('\n')
# write_text_path = '/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/simpleProDeal.txt'
write_text_path = '/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/confusion2.txt'
simShape = open('/Users/fighting/PycharmProjects/Text_error_detection/data/confusionSet/similarShape.txt', encoding='utf-8').read().split('\n')

def listTodict(cs, deli):
    ls = []
    for str in cs:
        print(str)
        sarr = str.split(deli)
        temp = []
        temp.append(sarr[0])
        temp.append(sarr[1])
        ls.append(temp)
    return dict(ls)
confusionM = listTodict(confusionSet, ":")
simShape = listTodict(simShape, ",")
print(simShape)
for key in simShape:
    print(simShape.get(key))
    if key in confusionM.keys():
        # print("key:"+key)
        value = simShape.get(key)
        varr = list(value)
        cv = confusionM.get(key)
        for v in varr:
            if v not in cv:
                # print("key1:"+cv)
                cv = cv+v
                confusionM[key] = cv
    else:
        print("不在困惑集中的数据")
        print(key+":"+simShape.get(key))
        confusionM[key] = simShape.get(key)
print(confusionM)

with open(write_text_path, 'w') as w:
    for key in confusionM:
        w.write(key+":"+confusionM.get(key))
        w.write("\n")
#
#         w.write('\n')