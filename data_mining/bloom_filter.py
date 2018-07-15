#coding:utf-8

import math
from dynamic_pybloom import BloomFilter

# darts: 10^9   targets=8*10^9

# one given target is hit probability
# false positive probability
def probability(m,n):

    #the probability that a bit remains 0
    return math.pow(math.e,-float(m)/n)

# false positive probability
def k_hash_function_probability(k,m,n):
    p = probability(k * m, n)
    return math.pow(1-p,k)
#
# for i in range(1,10):
#     print(k_hash_function_probability(i,1,8*i))

def bloom():
    bf = BloomFilter(80000, error_rate=0.1)
    for i in range(0,80000):
        bf.add(i)
    print(2 in bf)
    print(bf.count)

# bloom()

def hash_fun(a,b,x):
    return (a*x+b) % 32


def fm():
    x=[3,1,4,1,5,9,2,6,5]

    for i in x:
        for a,b in [(3,7)]:
            b=hash_fun(a,b,i)
            bin_b = bin(b)
            print((a,b,i), bin_b)




fm()






