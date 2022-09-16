# author:fighting
import re
def get_number(mystr):
    number = re.findall('\d+', mystr)[0]
    number = int(str(number))
    return number

num = get_number('小学同学有一位朋友的父亲是渔夫20')
print(num)