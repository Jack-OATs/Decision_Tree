import unittest
from decision_trees import information_gain

class InformationGainTest(unittest.TestCase):
    def test0(self):
        self.assertEqual(0, information_gain([0,0,0,0,0], [[0,0],[0,0,0]]))

    def test1(self):
        self.assertAlmostEqual(0.0817, information_gain([0, 0, 0, 1, 1, 1], [[0, 1, 0], [0, 0, 1]]), delta=0.00001)

    def test2(self):
        self.assertAlmostEqual(0.01997, information_gain([0, 1,0,1,0], [[0, 1], [0, 1, 0]]), delta=0.00001)

    def test3(self):
        self.assertAlmostEqual(0.97095, information_gain([0, 0, 1, 1, 1], [[0, 0], [1, 1, 1]]), delta=0.00001)