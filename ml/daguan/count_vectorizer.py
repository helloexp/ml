#coding:utf-8

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

texts=["dog cat fish","dog cat cat","fish bird", 'bird']

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()

# print((1,vectorizer.get_feature_names()))

cv_fit = vectorizer.fit_transform(texts)

print(2,vectorizer.get_feature_names())
print(3,cv_fit)
print(4,cv_fit.toarray())









