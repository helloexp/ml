# -*- coding:utf-8 -*-

import numpy as np


# https://zhuanlan.zhihu.com/p/24309547


a=[0,1,2,3,4,5]

# array
b = np.array(a)
print type(b)

print b
print b.shape
print b.argmax()
print b.max()


c = [[1, 2,3], [3, 4,5]]

d = np.array(c)
print d.shape

print d.size

print d.max(axis=0)
print d.max(axis=1)
print d.mean(axis=0)
print d.flatten()
print np.ravel(c)


# 3x3的浮点型2维数组，并且初始化所有元素值为1
e = np.ones((3, 3), dtype=np.float)

# 创建一个一维数组，元素值是把3重复4次，array([3, 3, 3, 3])
f = np.repeat(3, 4)


# 2x2x3的无符号8位整型3维数组，并且初始化所有元素值为0
g = np.zeros((2, 2, 3), dtype=np.uint8)
print g
print g.shape                    # (2, 2, 3)
h = g.astype(np.float)  # 用另一种类型表示


i=np.array([1,2])
j=np.array([3,4])

print np.dot(i,j)
print i+j


# linalg
# 线性代数 模块


a = np.array([3, 4])
print np.linalg.norm(a)

b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
c = np.array([1, 0, 1])

print "b*c=",np.dot(b,c)
print "b*c=",b*c

x = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
w=np.array([1,2])

print x,w
print np.dot(w.transpose(),x)

print w.shape
print w.transpose().shape

reverse = np.array([[2]])
print "reverse=",np.linalg.inv(reverse)