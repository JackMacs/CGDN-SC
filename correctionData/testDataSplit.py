test_text = open("/Users/fighting/PycharmProjects/Text_error_detection/correctionData/test_text", encoding="utf8").read().split("\n")
test_nolab = open("/Users/fighting/PycharmProjects/Text_error_detection/correctionData/test_nolab", encoding="utf8").read().split("\n")
test1, test2, test3, test4, test5 = [], [], [], [], []
testb1, testb2, testb3, testb4, testb5 = [], [], [], [], []
for i in range(0, 9113):
    test1.append(test_text[i])
    testb1.append(test_nolab[i])

for i in range(9114, 18055):
    test2.append(test_text[i])
    testb2.append(test_nolab[i])

for i in range(18056, 26986):
    test3.append(test_text[i])
    testb3.append(test_nolab[i])

for i in range(26987, 35018):
    test4.append(test_text[i])
    testb4.append(test_nolab[i])

for i in range(35019, 41246):
    test5.append(test_text[i])
    testb5.append(test_nolab[i])

with open("test_text1", "w") as wf:
    for str in test1:
        wf.write(str)
        wf.write("\n")

with open("test_text2", "w") as wf:
    for str in test2:
        wf.write(str)
        wf.write("\n")

with open("test_text3", "w") as wf:
    for str in test3:
        wf.write(str)
        wf.write("\n")

with open("test_text4", "w") as wf:
    for str in test4:
        wf.write(str)
        wf.write("\n")

with open("test_text5", "w") as wf:
    for str in test5:
        wf.write(str)
        wf.write("\n")

with open("test_lab1", "w") as wf:
    for str in testb1:
        wf.write(str)
        wf.write("\n")

with open("test_lab2", "w") as wf:
    for str in testb2:
        wf.write(str)
        wf.write("\n")

with open("test_lab3", "w") as wf:
    for str in testb3:
        wf.write(str)
        wf.write("\n")

with open("test_lab4", "w") as wf:
    for str in testb4:
        wf.write(str)
        wf.write("\n")

with open("test_lab5", "w") as wf:
    for str in testb5:
        wf.write(str)
        wf.write("\n")