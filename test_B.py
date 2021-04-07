import unittest
from decision_trees import function_B


class BTest(unittest.TestCase):
    def test0(self):
        self.assertEqual(0, function_B(0.0))

    def test1(self):
        self.assertEqual(0, function_B(1.0))

    def test2(self):
        self.assertEqual(1, function_B(0.5))

    def test3(self):
        self.assertAlmostEqual(0.81128, function_B(0.25), delta=0.00001)
