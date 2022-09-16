btrain = open('/Users/fighting/PycharmProjects/Text_error_detection/data/public_data/sighan2014/C1_training.txt', encoding='utf-8').read().split('\n')
train_text = []
train_correct = []
for i in range(len(btrain)):
    # print(btrain[i])
    if '<ESSAY ' in btrain[i]:
        i += 1
        str = btrain[i]
        temp1 = []
        temp2 = []
        while '</ESSAY' not in str:
            if '<PASSAGE id' in str:
                loc1 = str.index("=")
                loc2 = str.index(">")
                sNum = str[loc1+2:loc2-1]
                temp1.append(sNum)
                loc3 = str.index("</PASSAGE>")
                sTex = str[loc2+1:loc3]
                temp1.append(sTex)
                # print("sTex:"+sTex)
            if '<MISTAKE id' in str:
                loc1 = str.index("=")
                loc2 = str.index('" location')
                sNum = str[loc1+2:loc2]
                temp2.append(sNum)
                loc3 = str.index('on=')
                loc4 = str.index('">')
                sloc = str[loc3+4:loc4]
                # print("sloc:" + sloc)
                temp2.append(sloc)
            i += 1
            str = btrain[i]
        # print("temp1:")
        # print(temp1)
        # print("temp2:")
        # print(temp2)
        for j in range(0, len(temp1), 2):
            if temp1[j] == temp2[j]:
                loc = int(temp2[j+1])
                st = list(temp1[j+1])
                stemp = ""
                print("输入文本："+temp1[j+1])
                print("错误位置：", loc)
                for h in range(len(st)):
                    if st[h] in "；！，。？：」 「、）（？．":
                        if len(stemp) != 0:
                            train_text.append(stemp)
                        stemp = ""
                    if h == loc-1:
                        stemp = stemp + st[h]+"/e"
                        if h != len(st) - 1:
                            stemp = stemp + ' '
                    else:
                        if st[h] not in "；！，。？：」 「、）（？．":
                            stemp = stemp + st[h] + "/r"
                            if h != len(st) - 1:
                                stemp = stemp + ' '
                if(len(stemp)!= 0):
                    train_text.append(stemp)
print(train_text)


def writeFile(texts):
    with open('ctrain_sighan14.txt', 'w') as wf:
        for str in texts:
            wf.write(str)
            wf.write('\n')

writeFile(train_text)



