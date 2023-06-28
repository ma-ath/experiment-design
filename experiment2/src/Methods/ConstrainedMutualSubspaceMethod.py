import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import unittest
from itertools import groupby
from typing import List

from VectorSpace import VectorSpace
from VectorSet import VectorSet
from Methods.SubspaceMethod import SubspaceMethod

class ConstrainedMutualSubspaceMethod(SubspaceMethod):
    def __init__(self):
        raise NotImplementedError

# --- unittests
class TestConstrainedMutualSubspaceMethod(unittest.TestCase):
    def setUp(self):
        raise NotImplementedError

if __name__ == "__main__":
    unittest.main()