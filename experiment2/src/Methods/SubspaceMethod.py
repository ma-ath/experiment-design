import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import unittest
from typing import Tuple

from VectorSpace import VectorSpace
from VectorSet import VectorSet


class SubspaceMethod:
    """
    # 部分空間法
    
    Class that implements a simple subspace method for vector subspaces
    """
    def __init__(self, dim:int) -> None:
        if type(dim) is not int:
            raise(ValueError("VectorSpace dimension must be an integer"))
        if dim <= 0:
            raise(ValueError("VectorSpace dimension must be greater than 0"))

        self.dim = dim


    def train(self, data:np.ndarray, labels:list, min_energy:float=0.8) -> VectorSet:
        """
        Applies PCA to a set of subspaces
        """
        set = VectorSet(dim=self.dim)
        set.populate(data, labels)
        self.subset = set.pca(min_energy=min_energy)
        return self.subset

    def eval(self, data:np.ndarray, correct_labels:list) -> Tuple[list, float]:
        """
        Simple Method that implements a common evaluation procedure for classification problems
        Returns a list of correct classifications, and the success ratio
        """
        if data.ndim > 2:
            raise(AssertionError("Cannot input tensor of ndim > 2"))
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.shape[1] != self.dim:
            raise(AssertionError("Vector dimension must be the same as VectorSpace dimension"))  
        
        predicted_labels = self.classify(data)
        
        correct_class = []
        for l1, l2 in zip(predicted_labels, correct_labels):
            correct_class.append(l1 == l2)
        
        prediction_ratio = correct_class.count(True) / len(correct_class)
        
        return correct_class, prediction_ratio
    
    def classify(self, vectors:np.ndarray) -> float:
        """
        Classifies a ndarray in one of the subspaces using the cossine similarity
        """
        if vectors.ndim > 2:
            raise(AssertionError("Cannot input tensor of ndim > 2"))
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        if vectors.shape[1] != self.dim:
            raise(AssertionError("Vector dimension must be the same as VectorSpace dimension"))  
        
        # Classify all vectors by cossine similarity
        max_likelihood = [self.subset.labels[0]]*vectors.shape[0]
        cs = [0]*vectors.shape[0]

        for subspace in self.subset:
            foo = self.cossine_similarity(vectors, subspace)
            for i in range(len(foo)):
                if foo[i] > cs[i]: cs[i] = foo[i]; max_likelihood[i] = subspace.label
        return max_likelihood

    def cossine_similarity(self, vector:np.ndarray, subspace:VectorSpace) -> np.ndarray:
        """
        Returns S = \sum_{i=0}^{r-1} \frac{(x,\phi_i)^2}{\|x\|\|\phi_i\|}
        """
        if vector.ndim > 2:
            raise(AssertionError("Cannot input tensor of ndim > 2"))
        if vector.ndim == 1:
            vector = vector[np.newaxis, :]
        if vector.shape[1] != self.dim:
            raise(AssertionError("Vector dimension must be the same as VectorSpace dimension"))       

        vector = vector.astype(subspace.dtype)

        S = np.sum(
                np.divide(
                    np.matmul(vector, subspace.A.transpose())**2,
                    np.matmul(
                        np.sqrt(
                            np.diag(
                                np.matmul(vector, vector.transpose()
                                )
                            )
                        )[np.newaxis, :].transpose(),
                        np.sqrt(
                            np.diag(
                                np.matmul(subspace.A, subspace.A.transpose())
                            )
                        )[np.newaxis, :]
                    )
                ), axis=1
            )
        return S


# --- unittests
class TestVectorSM(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 10
        self.dim = 32
        self.loop = 20
        return super().setUp()
    
    def test_init(self):
        with self.assertRaises(TypeError):
            SubspaceMethod()
        with self.assertRaises(ValueError):
            SubspaceMethod(dim = -1)
        with self.assertRaises(ValueError):
            SubspaceMethod(dim = 0)
        with self.assertRaises(ValueError):
            SubspaceMethod(dim = "Hello")
        SubspaceMethod(dim = self.dim)

    def test_train(self):
        sm = SubspaceMethod(dim=self.dim)
        mock_data = np.random.rand(10*self.n, self.dim)
        mock_labels = [i%10 for i in list(range(100))]
        _ = sm.train(mock_data, mock_labels)

    def test_cossine_similarity(self):
        sm = SubspaceMethod(dim=2)
        subspace = VectorSpace(dim=2)
        subspace.append(np.array([1, 0]))
        vector = np.array([0, 1])
        self.assertTrue(np.allclose(sm.cossine_similarity(vector, subspace), np.zeros(1)))
        vector = np.array([1, 0])
        self.assertTrue(np.allclose(sm.cossine_similarity(vector, subspace), np.ones(1)))
        vector = np.array([[0, -2], [1, 0]])
        self.assertTrue(np.allclose(sm.cossine_similarity(vector, subspace), np.array([0, 1])))
    
    def test_classify(self):
        n = 10
        classes = 10
        sm = SubspaceMethod(dim=self.dim)
        mock_data = np.random.rand(n*classes, self.dim)
        mock_labels = [i%classes for i in list(range(n*classes))]
        sm.train(mock_data, mock_labels)
        mock_vector = np.random.rand(classes, self.dim)
        labels = sm.classify(mock_vector)
        self.assertEqual(len(labels), classes)
    
    def test_eval(self):
        n = 10
        classes = 10
        sm = SubspaceMethod(dim=self.dim)
        mock_data = np.random.rand(n*classes, self.dim)
        mock_labels = [i%classes for i in list(range(n*classes))]
        sm.train(mock_data, mock_labels)
        mock_vector = np.random.rand(n, self.dim)
        mock_labels = [i for i in list(range(classes))]
        eval = sm.eval(mock_vector, mock_labels)


if __name__ == "__main__":
    unittest.main()