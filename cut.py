import re
with open('./data/public_simple_data/rightSen15', encoding='utf-8') as f:
    for txt in f:
        txt = txt.strip()
        pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
        result_list = re.split(pattern, txt)
        with open('./data/public_simple_data/right15cut', 'a', encoding='utf-8') as file:
            for j in range (len(result_list)):
                file.write(result_list[j]+'\n')
                print(result_list[j])