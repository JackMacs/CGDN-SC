b_sbds = open("/Users/fighting/PycharmProjects/Text_error_detection/correctionData/btrain_sighan14.txt", encoding="utf-8").read().split("\n")
test = open("/Users/fighting/PycharmProjects/Text_error_detection/data/public_simple_data/test1_lab", encoding="utf-8").read().split("\n")
c_sbds = open("/Users/fighting/PycharmProjects/Text_error_detection/correctionData/ctrain_sighan14.txt", encoding="utf-8").read().split("\n")
def count(dataset):
    n = 0;
    for ds in dataset:
        if "e" in ds:
            n += 1
    return n

print(count(b_sbds))
print(count(c_sbds))
print(count(test))