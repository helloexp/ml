# coding:utf-8


# http://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E6%88%91%E7%9A%84%E9%98%9F%E4%BC%8D.html

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import sys
import csv

csv.field_size_limit(sys.maxsize)


df_train = pd.read_table("/Users/tong/Downloads/study/watermelon/new_data/train_set.csv",sep=",",chunksize=10**6)
print(df_train)


# id,article,word_seg,class
df_train.drop(["article", "id"],axis=1,inplace=True)

df_test = pd.read_csv(open("/Users/tong/Downloads/study/watermelon/new_data/test_set.csv","rU"),engine="python")
df_test.drop(["article"],axis=1,inplace=True)

# print(df_train)
#what meaning
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000)
vectorizer.fit(df_train["word_seg"])
x_train = vectorizer.transform(df_train["word_seg"])
x_test = vectorizer.transform(df_test["word_seg"])

y_train = df_train["class"]-1

# print(y_train)

lg = LogisticRegression(C=4, dual=True)
lg.fit(x_train,y_train)

y_test = lg.predict(x_test)
df_test["class"]=y_test.tolist()

df_test["class"]=df_test["class"]+1

df_result=df_test.loc[:,["id","class"]]

df_result.to_csv("./result.csv",index=False)










