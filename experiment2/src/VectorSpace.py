import numpy as np
import unittest
from typing import Tuple


class VectorSpace:
    """
    Class that defines a simple subspace
    """
    def __init__(self, dim:int, label=None) -> None:
        """
        Initialize a vector space of dimension `dim`
        Args:
            `dim`: dimension of vector space
        """
        if type(dim) is not int:
            raise(ValueError("VectorSpace dimension must be an integer"))
        if dim <= 0:
            raise(ValueError("VectorSpace dimension must be greater than 0"))

        self.dtype = np.float32
        self.n = 0
        self.singular_values = None
        self.dim = dim
        self.label = label
        self.A = np.empty([self.n, self.dim])
    
    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i:int) -> np.ndarray:
        if i < self.n and i >= 0:
            return self.A[i]
        raise(IndexError("Index i out of bound"))

    def append(self, vector:np.ndarray, singular_values:list = None) -> None:
        if vector.ndim > 2:
            raise(AssertionError("Cannot input tensor of ndim > 2"))
        if vector.ndim == 1:
            vector = vector[np.newaxis, :]
        if vector.shape[1] != self.dim:
            raise(AssertionError("Vector dimension must be the same as VectorSpace dimension"))
        if singular_values is not None:
            if type(singular_values) is not list:
                singular_values = [singular_values]
            if len(singular_values) != vector.shape[0]:
                raise(AssertionError("List of singular values must have the same batch lenght as vector"))
        else:
            singular_values = [1]*vector.shape[0]

        vector = vector.astype(self.dtype)
        self.n = self.n + vector.shape[0]
        self.A = np.concatenate((self.A, vector), axis=0)

        if self.singular_values is None:
            self.singular_values = singular_values
        else:
            self.singular_values.append(singular_values)
    
    def remove(self, n:int) -> None:
        if n >= self.n:
            raise(ValueError(f"Cannot remove vector from position {n}. VectorSpace has {self.n} vectors."))
        self.A = np.delete(self.A, n, axis=0)
        self.n -= 1
        self.singular_values.pop(n)

    def svd(self, A:np.ndarray=None, full_matrices:bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if A is None:
            A = self.A
        U, S, Vh = np.linalg.svd(A, full_matrices=full_matrices)
        return U, S, Vh

    def pca(self, min_energy:float=0.8, standardize:bool = False) -> 'VectorSpace':
        # calculate mean vector
        A = self.A

        if standardize:
            mean_vector = A.mean(axis=0)
            A = np.subtract(A, mean_vector)
            A = np.divide(A, np.sqrt(self.dim))

        # Calculate SVD
        U, S, Vh = self.svd(A, full_matrices=False)

        # Get base vectors for subspace from min_energy
        if min_energy == 1:
            n = self.n
        else:
            cumulative_energy = np.cumsum(S, axis=0) / np.sum(S)
            for i, energy in enumerate(cumulative_energy):
                n = i+1
                if energy >= min_energy:
                    break
        # Generate Subspace
        subspace = VectorSpace(dim=self.dim)
        subspace.append(Vh[:n], singular_values=S.tolist()[:n])

        return subspace

    def __lt__(self, other:'VectorSpace') -> bool:
        if other.dim != self.dim:
            raise(AssertionError("Cannot compare two subspaces with different dimensions"))

        return self.n < other.n
    
    def __str__(self):
        return f"VectorSpace:{self.n}x{self.dim}"

    def __repr__(self):
        return f"VectorSpace:{self.n}x{self.dim}"

    def __sub__(self, other:'VectorSpace') -> 'VectorSpace':
        if type(other) is not VectorSpace:
            raise(SyntaxError(f"Cannot perform subtraction of {type(self)} and {type(other)}"))
        if other.dim != self.dim:
            raise(AssertionError(f"Both VectorSpaces must have same dimension"))
        if other.n != self.n:
            raise(AssertionError(f"Both VectorSpaces must have same number of vectors"))

        DS = VectorSpace(dim=self.dim)
        DS.append(self.A - other.A)
        return DS

# --- unittests
class TestVectorSpace(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 10
        self.dim = 32
        self.loop = 20
        return super().setUp()

    def test_init(self):
        with self.assertRaises(TypeError):
            VectorSpace()
        with self.assertRaises(ValueError):
            VectorSpace(dim = -1)
        with self.assertRaises(ValueError):
            VectorSpace(dim = 0)
        with self.assertRaises(ValueError):
            VectorSpace(dim = "Hello")
        VectorSpace(dim = self.dim)

    def test_getitem_len(self):
        subspace = VectorSpace(dim=self.dim)
        subspace.append(np.random.rand(self.n, self.dim))
        self.assertEqual(subspace[0].shape[0], self.dim)
        self.assertEqual(len(subspace), self.n)
        with self.assertRaises(IndexError):
            subspace[11]
    
    def test_svd(self):
        subspace = VectorSpace(dim=self.dim)
        subspace.append(np.eye(self.dim))
        U, S, V = subspace.svd()
        self.assertTrue(np.allclose(S, np.ones(self.dim)))
        eye_A = np.eye(self.dim, self.dim)
        U, S, V = subspace.svd(eye_A)
        self.assertTrue(np.allclose(S, np.ones(self.dim)))
        ones_A = np.ones([self.dim, self.dim])
        U, S, V = subspace.svd(ones_A)
        self.assertTrue(np.allclose(S, np.concatenate(([self.dim], np.zeros(self.dim-1)))))

    def test_append(self):
        subspace = VectorSpace(dim=self.dim)
        vector_1 = np.random.rand(self.dim)
        subspace.append(vector_1)
        self.assertTrue(np.allclose(subspace[0], vector_1))

        subspace = VectorSpace(dim=self.dim)
        n = 10
        vector_n = np.random.rand(n, self.dim)
        subspace.append(vector_n)
        for i in range(n):
            self.assertTrue(np.allclose(subspace[i], vector_n[i]))
    
    def test_remove(self):
        subspace = VectorSpace(dim=self.dim)
        with self.assertRaises(ValueError):
            subspace.remove(0)

        vector_1 = np.random.rand(self.dim)
        subspace.append(vector_1)
        self.assertTrue(len(subspace) == 1)
        subspace.remove(0)
        self.assertTrue(len(subspace) == 0)

        subspace = VectorSpace(dim=self.dim)
        n = 10
        vector_n = np.random.rand(n, self.dim)
        subspace.append(vector_n)
        subspace.remove(n-1)
        self.assertTrue(np.allclose(subspace.A, vector_n[:n-1, :]))

    def test_pca(self):
        for _ in range(self.loop):
            rand_data = np.eye(self.dim)+np.random.rand(self.dim, self.dim)

            subspace = VectorSpace(dim=self.dim)
            subspace.append(rand_data)
            pca = subspace.pca()
            self.assertTrue(len(pca) <= len(subspace))
            pca = subspace.pca(standardize=True)
            self.assertTrue(len(pca) <= len(subspace))
            
            pca = subspace.pca(min_energy=1)
            self.assertTrue(len(pca) == len(subspace))
            pca = subspace.pca(min_energy=1, standardize=True)
            self.assertTrue(len(pca) == len(subspace))
    
    def test_lt(self):
        s1 = VectorSpace(dim=self.dim)
        s2 = VectorSpace(dim=self.dim)
        s3 = VectorSpace(dim=self.dim)
        s4 = VectorSpace(dim=self.dim)
        s5 = VectorSpace(dim=self.dim)

        s1.append(np.random.rand(1, self.dim))
        s2.append(np.random.rand(2, self.dim))
        s3.append(np.random.rand(3, self.dim))
        s4.append(np.random.rand(4, self.dim))
        s5.append(np.random.rand(5, self.dim))

        foo_list = [s4, s3, s5, s2, s1]
        self.assertEqual(sorted(foo_list), [s1, s2, s3, s4, s5])

    def test_sub(self):
        with self.assertRaises(SyntaxError):
            _ = VectorSpace(dim=2) - np.random.rand(2,2)
        with self.assertRaises(AssertionError):
            _ = VectorSpace(dim=2) - VectorSpace(dim=3)

        subspace1 = VectorSpace(dim=self.dim)
        subspace2 = VectorSpace(dim=self.dim)
        subspace1.append(np.eye(self.dim)[int(np.floor(self.dim/2)):self.dim])

        with self.assertRaises(AssertionError):
            _ = VectorSpace(dim=self.dim) - subspace1

        subspace2.append(np.eye(self.dim)[0:int(np.floor(self.dim/2))])
        _ = subspace1 - subspace2

if __name__ == "__main__":
    unittest.main()