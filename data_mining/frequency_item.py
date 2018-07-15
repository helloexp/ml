# coding:utf-8
import os
import numpy as np


def exe_100_100():
    bucket = range(1, 101)

    res = []
    for i in bucket:
        tmp = []
        for item in range(1, i + 1):
            if (i % item == 0):
                tmp.append(item)
        res.append(tmp)
    return res


def trinagular_matrix(arr, i, j, ele):
    index = int((i - 1) * (len(arr) - i / 2) + j - i)
    # index=int(i*(i+1)/2)+j
    print(index)
    arr[index - 1] = ele


class APrior(object):
    def __init__(self, datas, k):
        self.k = k
        self.datas = datas

    def find(self):

        count_dict = APrior.word_count(self.datas)

        for i in range(2, self.k):
            if i == 2:
                two_possible = filter(lambda w, c: c >= i, count_dict)
                k_paris=APrior.generate_pairs_by_k(two_possible,i)

    @staticmethod
    def distinct(datas):
        res = set()
        for data in datas:
            for d in data:
                res.add(d)
        return res

    @staticmethod
    def word_count(datas):
        distinct = APrior.distinct(datas)
        count_dict = {}
        for d in distinct:
            if d in count_dict:
                c = count_dict.get(d) + 1
                count_dict.update({d: c})
            else:
                count_dict.update({d: 0})

        return count_dict

    @staticmethod
    def generate_pairs_by_k(arr, k):
        if len(arr)<k:
            return None
        elif len(arr)==k:
            return [arr]
        else:
            res=[]
            for i in range(0,len(arr)):
                first = arr[i]
                for kk in range(0,k-1):
                    k_1_pair = APrior.generate_pairs_by_k(arr[i + 1:], kk)


if __name__ == '__main__':
    res = exe_100_100()
    print(res)
    # arr = np.zeros((3))
    # trinagular_matrix(arr, 2, 3, 5)
    # print(arr)
