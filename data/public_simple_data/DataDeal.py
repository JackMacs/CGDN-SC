test_text = open("/Users/fighting/PycharmProjects/Text_error_detection/data/public_simple_data/test1_text", encoding="utf-8").read().split("\n")
test_lab = open("/Users/fighting/PycharmProjects/Text_error_detection/data/public_simple_data/test1_lab", encoding="utf-8").read().split("\n")
def text_split_lab(text, lab):
    te = []
    for sen in text:
        temp = sen.split(lab)
        for t in temp:
            te.append(t)
    return te

print(test_text)
te = text_split_lab(test_text, "；")
# te = text_split_lab(te, "、")
# te_lab = text_split_lab(test_text, "；/")
te_lab1 = []
te_lab2 = []
for sen in test_lab:
    temp = []
    if "；/e" in sen:
        temp = sen.split("；/e")
    if "；/r" in sen:
        temp = sen.split("；/r")
    else:
        temp.append(sen)
    for t in temp:
        te_lab1.append(t)

# for sen in te_lab1:
#     temp = []
#     if "、/e" in sen:
#         temp = sen.split("、/e ")
#     if "、/r" in sen:
#         temp = sen.split("、/r ")
#     else:
#         temp.append(sen)
#     for t in temp:
#         te_lab2.append(t)
print("te")
print(te)
print(te_lab2)

with open("/Users/fighting/PycharmProjects/Text_error_detection/correctionData/test_text_nosign", "w") as wf:
    for t in te:
        wf.write(t)
        wf.write("\n")

with open("/Users/fighting/PycharmProjects/Text_error_detection/correctionData/test_lab_nosign", "w") as wf:
    for t in te_lab1:
        wf.write(t)
        wf.write("\n")


