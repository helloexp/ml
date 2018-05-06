import math
import random


def line_function(x):
    error = random.randint(-10, 10) * (x * 0.025)
    return 10*x+1 +error


def generate_function_data(func,num=10):

    range_x = range(1, num)
    data = map(func, range_x)

    return range_x, data

if __name__ == '__main__':
    print generate_function_data(line_function)

