import unittest

from data_mining.jaccard import *


class Jaccard(unittest.TestCase):
    def test_jaccard(self):
        s1 = [1, 2, 3, 4]
        s2 = [2, 3, 5, 7]

        jac = jaccard(s1, s2)

        self.assertEqual(1 / 3, jac)

    def test_word_count(self):
        s = [1, 2, 3, 2]
        s_count_dic = word_count(s)
        count_1 = s_count_dic.get(1)
        count_2 = s_count_dic.get(2)

        self.assertEqual(count_1, 1)
        self.assertEqual(count_2, 2)

    def test_jaccard_bag(self):
        s1 = [1, 1, 7, 2]
        s2 = [1, 2, 3, 4, 5]

        jac_bag = jaccard_bag(s1, s2)

        self.assertEqual(1 / 3, jac_bag)

    def test_matrix_jaccard(self):
        a=[0,1,1,0,1,0]
        b=[1,0,1,0,1,1]

        jacard = martix_jacard(a, b)

        self.assertEqual(0.4, jacard)



