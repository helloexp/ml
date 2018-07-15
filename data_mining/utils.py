import math


def find_prime(num):
    math_sqrt = math.sqrt(num)

    sqrt = int(math_sqrt) + 1

    for i in range(2, sqrt):
        if (num % i == 0):
            return False

    return True
