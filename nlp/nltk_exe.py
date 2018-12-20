# coding=utf-8

import nltk
from nltk.corpus import brown as cb
from nltk.corpus import gutenberg as cg

print(dir(cb))
print(cb.categories()) #返回语料库包含的类别
print(cb.fileids())  #返回语料库文件
print(cb.words()) #显示语料库中的单词
# print(cb.raw())
print(cb.sents()) #句子
print(cb.tagged_words())

# print(cg.fileids())

# print(cb.fileids(["cc12"]))
print(len(cb.words(fileids=['cc12'])))



print(dir(nltk.corpus)) #打印语料库的名字


from nltk.stem import PorterStemmer
word="unexpected"

port=PorterStemmer()
stem = port.stem(word)

print(stem)
print(port.stem(stem))


