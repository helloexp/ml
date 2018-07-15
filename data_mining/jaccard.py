import numpy as np


def jaccard(s1, s2):
    s1 = set(s1)
    s2 = set(s2)

    union = s1.union(s2)

    if (len(union) == 0):
        return 0

    inter = s1.intersection(s2)
    return len(inter) / len(union)

def martix_jacard(arr1,arr2):
    if(len(arr1)!=len(arr2)):
        return 0

    zip_arr = zip(arr1, arr2)

    element=0
    denominator=0
    for x,y in zip_arr:
        if x==y==1:
            element+=1
        if x==1 or y==1:
            denominator+=1

    if denominator==0:
        return 0
    else:
        return element/denominator





def word_count(s1):
    r = {}
    for s in s1:
        s_count = r.get(s, 0)
        s_count += 1
        r.update({s: s_count})
    return r


def jaccard_bag(s1, s2):
    s1_count = word_count(s1)
    s2_count = word_count(s2)

    s1_keys = set(s1_count.keys())
    s2_keys = set(s2_count.keys())

    inter = s1_keys.intersection(s2_keys)

    s_count = 0
    for i in inter:
        get1 = s1_count.get(i)
        get2 = s2_count.get(i)
        s_count += max(get1, get2)

    all_len = len(s1) + len(s2)
    return s_count / all_len
