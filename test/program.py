# -*- coding:utf-8 -*-i
import math


def closest_power(base,num):

    log = math.log(num, base)

    if(log % 1 < 0.5):
        return math.floor(log)
    else:
        return math.ceil(log)


def dotProduct(listA,listB):

    zipList = zip(listA, listB)

    l = map(lambda x: x[0]* x[1], zipList)
    return sum(l)

def dotProduct2(listA,listB):
    sum=0
    for i in range(0,len(listA)):
        sum=sum+listA[i]*listB[i]

    return sum

def is_triangular(k):
    s=0
    for i in range(0,k/2):
        s=s+i
        if(s==k):
            return True

    return False

if __name__ == '__main__':

    print(closest_power(3,12))
    print (closest_power(4,12))
    print (closest_power(4,1))

    print (dotProduct([1,2,3],[4,5,6]))
    print (dotProduct2([1,2,3],[4,5,6]))

    print (is_triangular(10))


