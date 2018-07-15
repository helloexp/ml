import unittest

from data_mining.utils import find_prime


class PrimeTest(unittest.TestCase):

    def test_prime(self):
        x=[1,2,3,4,6,8,9,10,11,12,13]
        # x=[2,4]

        for i in x:
            prime = find_prime(i)
            print(i,prime)








