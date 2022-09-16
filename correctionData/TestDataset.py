import re
import numpy as np
sens = open("/Users/fighting/PycharmProjects/Text_error_detection/correctionData/candidates1.txt").read().split("\n")
text = []
for sen in sens:
    sarr = sen.split("@")
    for st in sarr:
        if "/r" in st:
            print(st)
            text.append(st)


test_nolab = []
# 获取带有标签的测试文本对应的无标签测试文本
for sentence in text:
    words = []
    # print("sentence:", sentence)
    groups = re.findall('(.)/(.)', sentence)
    arrays = np.asarray(groups)
    words.append(arrays[:, 0])
    # print("words:", words)
    temp = words[0]
    tsr = ""
    for t in temp:
        tsr = tsr + t
    print("words:", tsr)

    test_nolab.append(tsr)


with open("test_text", "w") as wf:
    for st in text:
        wf.write(st)
        wf.write("\n")

with open("test_nolab", "w") as wf1:
    for st in test_nolab:
        wf1.write(st)
        wf1.write("\n")