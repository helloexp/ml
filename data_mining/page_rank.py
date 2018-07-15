# coding:utf-8

import numpy as np

# m = np.asarray([[1 / 2, 1 / 2, 0], [1 / 2, 0, 1], [0, 1 / 2, 0]])
# m = np.asarray([[1 / 3, 1 / 2, 0], [1 / 3, 0, 1/2], [1/3, 1 / 2, 1/2]])
m = np.asarray([[0, 1 / 3, 1 / 4, 1 / 4, 1/5],
                [1 / 3, 0, 1 / 4, 1 / 4, 1/5],
                [1 / 3, 1 / 3, 0, 1 / 4, 1/5],
                [0, 0, 1 / 4, 0, 1/5],
                [1 / 3, 1 / 3, 1 / 4, 1 / 4, 1/5]])

x = (1 / 5) * np.ones((5))

epsilo = 0.01
beta = 0.8


def page_rank_al(m, x):
    # new_x = beta* m.dot(x) + (1-beta)*1/len(x)
    new_x = m.dot(x)
    print(new_x)
    if np.linalg.norm(x - new_x) > epsilo:
        return page_rank_al(m, new_x)
    else:
        return new_x


# print(page_rank_al(m, x))


L=np.array([[0,1,1,1,0],[1,0,0,1,0],[0,0,0,0,1],[0,1,1,0,0],[0,0,0,0,0]])
h = np.ones((5))

a=L.transpose().dot(h)


def normalization(a):
    max = np.max(a)
    return a/max

a=normalization(a)

def HITS(L,h,a):
    new_h=normalization(L.dot(a))

    new_a = normalization(L.transpose().dot(h))

    if np.linalg.norm(h-new_h)>epsilo or np.linalg.norm(a-new_a)>epsilo:
        return HITS(L,new_h,new_a)
    else:
        return new_h,new_a


h_res,a_res = HITS(L, h, a)
print(h_res)
print(a_res)
