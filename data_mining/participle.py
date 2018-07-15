# encoding=utf-8
import os

import jieba

base_dir="../resource/text/"

text_file = os.listdir(base_dir)


def cut_words(param):

    res=[]

    for p in param:
        cut = jieba.cut(p, cut_all=False)
        res.append(cut)
    return res


for f in text_file:
    text = open(base_dir+f,"r")

    words = cut_words(text.readlines())
    for w in words:
        print("/ ".join(w))
    text.close()

