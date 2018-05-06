

def triangle(a,b,c):
    b_c = a + b > c
    if(b_c):
        return "sure"
    else:
        return "can not"

def add(a,b):
    return a+b

if __name__ == '__main__':

    #1,2,3
    res = triangle(0,1,0)
    print(res)




