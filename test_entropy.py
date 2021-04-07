import unittest
from decision_trees import entropy

class EntropyTest(unittest.TestCase):
    def test0(self):
        self.assertEqual(0, entropy([0, 0, 0, 0, 0]))

    def test1(self):
        self.assertEqual(0, entropy([1, 1, 1, 1, 1]))

    def test2(self):
        self.assertEqual(1, entropy([0, 0, 0, 1, 1, 1]))

    def test3(self):
        self.assertAlmostEqual(0.97095, entropy([0, 1, 0, 1, 0]), delta=0.00001)