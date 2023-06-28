import numpy as np
import unittest
from itertools import groupby
import random

from VectorSpace import VectorSpace


class VectorSet:
    """
    Class that defines a set of VectorSpaces
    """
    def __init__(self, dim:int, labels:list=[]) -> None:
        """
        labels (list): list of labels for each vspace
        dim (int): size of vector in each vspace
        """
        if type(dim) is not int:
            raise(ValueError("VectorSpace dimension must be an integer"))
        if dim <= 0:
            raise(ValueError("VectorSpace dimension must be greater than 0"))

        self.labels = list(labels)
        self.dim = dim
        # In this version, we make all subspaces have the same dim size.
        # Maybe this is not completely necessary
        self.set = {label: VectorSpace(dim=self.dim, label=label) for label in self.labels}
    
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, label) -> VectorSpace:
        if label in self.labels:
            return self.set[label]
        raise(IndexError(f"Label {label} not in VectorSet"))
    
    def populate(self, vectors: np.ndarray, labels:list, singular_values:list=None) -> None:
        """
        Populates VectorSpaces with vectors.
        If label does not exists in label list, generate it
        """
        if vectors.ndim > 2:
            raise(AssertionError("Cannot input tensor of ndim > 2"))
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        if vectors.shape[1] != self.dim:
            raise(AssertionError("Vector dimension must be the same as VectorSpace dimension"))       
        if len(vectors) != len(labels):
            raise(AssertionError("Each column vector must have a corresponding label in list"))
        if singular_values is not None:
            if type(singular_values) is not list:
                singular_values = [singular_values]
            if len(singular_values) != vectors.shape[0]:
                raise(AssertionError("List of singular values must have the same batch lenght as vector"))
        else:
            singular_values = [1]*vectors.shape[0]

        # Order labels and vector
        tensor_list = vectors.tolist()
        lt = zip(labels, tensor_list, singular_values)
        lt = sorted(lt)
        sorted_tensor = np.array([i for _, i, _ in lt])
        sorted_labels = [i for i, _, _ in lt]
        sorted_singular_values = [i for _, _, i in lt]

        # Group labels
        group_list = [list(g) for _, g in groupby(sorted_labels)]

        # Populate subspaces in batches
        i = 0
        for group in group_list:
            label = group[0]
            if label not in self.labels:
                self.labels.append(label)
                self.set[label] = VectorSpace(dim=self.dim, label=label)

            self.set[label].append(sorted_tensor[i:i+len(group)], sorted_singular_values[i:i+len(group)])
            i += len(group)

    
    def pca(self, min_energy:float=0.8) -> None:
        subset = VectorSet(labels=self.labels, dim=self.dim)
        for label, vspace in self.set.items():
            vsubspace = vspace.pca(min_energy)
            subset.populate(vsubspace.A, [label]*len(vsubspace), singular_values = vsubspace.singular_values)
        return subset

    def __str__(self) -> str:
        return f"VectorSet:{[str(self.set[label]) for label in self.labels]}"

# --- unittests
class TestSubspaces(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 10
        self.dim = 32
        self.loop = 20
        return super().setUp()
    
    def test_init(self):
        with self.assertRaises(TypeError):
            VectorSet()
        with self.assertRaises(ValueError):
            VectorSet(dim = -1)
        with self.assertRaises(ValueError):
            VectorSet(dim = 0)
        with self.assertRaises(ValueError):
            VectorSet(dim = "Hello")
        VectorSet(dim = self.dim)
  

    def test_getitem_len(self):
        set = VectorSet(dim=self.dim, labels=list(range(10)))
        self.assertEqual(len(set), 10)
        with self.assertRaises(IndexError):
            s = set[11]
        s = set[9]
    
    def test_populate(self):
        n_labels = 10
        n_vectors = 20
        n_repetitions = 10

        set = VectorSet(dim=self.dim)
        slow_set = VectorSet(dim=self.dim)

        mock_labels = list(range(n_labels))*n_repetitions
    
        for _ in range(n_vectors):
            random.shuffle(mock_labels)
            mock_data = np.random.rand(n_labels*n_repetitions, self.dim)           
            set.populate(mock_data, mock_labels)
            slow_set = self.populate_slow_implementation(slow_set, mock_data, mock_labels)

        self.assertTrue(len(set.labels) == n_labels)
        self.assertTrue(len(set[0]) == n_vectors*n_repetitions)
        
        # Since VectorSet sorts vectors, the order from the slow_set is not necessarily maintained.
        # Therefore, look for a similar vector in slow_set and see if can find it in the set
        for i in range(n_labels):
            for j in range(n_vectors*n_repetitions):
                m = 0
                for k in range(n_vectors*n_repetitions-m):
                    if np.allclose(set[i][j], slow_set[i][k]):
                        slow_set[i].remove(k)
                        break
                    m = m + 1
                    if k == n_vectors*n_repetitions-m:
                        self.assertTrue(False)

        with self.assertRaises(AssertionError):
            mock_data = np.random.rand(10, self.dim)
            mock_labels = list(range(9))
            set.populate(mock_data, mock_labels)

        with self.assertRaises(AssertionError):
            mock_data = np.random.rand(10, 24)
            mock_labels = list(range(10))
            set.populate(mock_data, mock_labels)

    def test_pca(self):
        n = 100
        n_classes = 10
        set = VectorSet(dim=self.dim)
        mock_data = np.random.rand(n, self.dim)
        mock_labels = [i%n_classes for i in list(range(n))]
        set.populate(mock_data, mock_labels)
        subset = set.pca()

    def populate_slow_implementation(self, vector_set:VectorSet, vectors:np.ndarray, labels:list) -> VectorSet:
        return_set = vector_set

        for vector, label in zip(vectors, labels):
            # Check if label exists. If not, generate new vspace
            if label not in return_set.labels:
                return_set.labels.append(label)
                return_set.set[label] = VectorSpace(dim=return_set.dim, label=label)
            return_set.set[label].append(vector)
        return return_set

    def test_str(self):
        set = VectorSet(dim=self.dim)
        s = str(set)

if __name__ == "__main__":
    unittest.main()