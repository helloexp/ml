
import unittest

from data_mining.clusting import generate_shingles


class Clusting(unittest.TestCase):

    def test_shingle(self):
        content="I am a very good boy"

        shingles = generate_shingles(2, content)

        print(shingles)



