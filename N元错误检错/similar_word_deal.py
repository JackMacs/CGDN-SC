# author:fighting
import re
similarWords = open('data/similarWords').read().split('\n')
with open('data/similarWords_deal', 'w') as w:
    for str in similarWords:
        temp = re.split(',', str)
        print(temp[1]+":"+temp[3])
        w.write(temp[1]+":"+temp[3])
        w.write('\n')
