# -*- coding:utf-8 -*-i

def generate():

    for i in range(101,1000):
        i

def dec2bin(string_num):
    num = int(string_num)
    mid = []
    while True:
        if num == 0: break
        num,rem = divmod(num, 2)
        mid.append(base[rem])

    return ''.join([str(x) for x in mid[::-1]])

base = [str(x) for x in range(10)] + [ chr(x) for x in range(ord('A'),ord('A')+6)]

if __name__ == '__main__':
    print(dec2bin(101))
    print(bin(101))







