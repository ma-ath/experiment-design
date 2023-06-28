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

class MutualSubspaceMethod(SubspaceMethod):
    """
    # 相互部分空間法

    Class that implements a simple mutual subspace method for vector subspaces
    """
    def eval(self, data: List[VectorSpace], correct_labels:list):
        """
        Simple Method that implements a common evaluation procedure for classification problems
        Returns a list of correct classifications, and the success ratio
        """
        if type(data) is not list:
            data = [data]
        assert(type(data[0] == VectorSpace))
        assert(len(data) == len(correct_labels))
        
        predicted_labels = self.classify(data)
        
        correct_class = []
        for l1, l2 in zip(predicted_labels, correct_labels):
            correct_class.append(l1 == l2)
        
        prediction_ratio = correct_class.count(True) / len(correct_class)
        
        return correct_class, prediction_ratio
    
    def classify(self, vspaces: List[VectorSpace]):
        """
        Classifies a list of subspace in one of the subspaces using the subspace_similarity metric
        """
        # Check if passing a list or a VectorSpace. Correct if necessary
        if type(vspaces) is not list:
            vspaces = [vspaces]
        assert(type(vspaces[0] == VectorSpace))

        # Classify all vectors by subspace similarity
        max_likelihood = [self.subset.labels[0]]*len(vspaces)
        ss = [0]*len(vspaces)

        for subspace in self.subset:
            foo = self.subspace_similarity(vspaces, subspace)
            for i in range(len(foo)):
                if foo[i] > ss[i]: ss[i] = foo[i]; max_likelihood[i] = subspace.label
            
        return max_likelihood

    def subspace_similarity(self, vspaces: List[VectorSpace], subspace:VectorSpace):
        cossine_similarities = self.cossine_similarity(vspaces, subspace)
        S = np.sum(cossine_similarities, axis=1) / cossine_similarities.shape[1]
        return S
        
    def cossine_similarity(self, vspaces: List[VectorSpace], subspace:VectorSpace) -> np.ndarray:
        """
        Returns the cossine similarity between a list of vector spaces and one subspace
        return shape: [vspace, cos^2_i]
        """
        if type(vspaces) is not list:
            vspaces = [vspaces]
        if type(vspaces[0]) != VectorSpace:
            raise(SyntaxError(f"You must pass a VectorSpace, not {type(subspace)}"))

        # Order list of vector spaces and group vector spaces of same size in batches
        original_order = np.linspace(1, len(vspaces), len(vspaces)).tolist()

        nlist = [space.n for space in vspaces]
        z = zip(vspaces, nlist, original_order)
        z = sorted(z)

        sorted_vspaces = []
        sorted_nlist = []
        new_order = []

        for vs, nl, no in z:
            sorted_vspaces.append(vs)
            sorted_nlist.append(nl)
            new_order.append(no)

        group_nlist = [len(list(n)) for _, n in groupby(sorted_nlist)]

        # Group vector spaces in 3d ndarrays, calculate cossine similarities in batches
        i = 0
        cossine_similarities = []
        for n in group_nlist:
            vspace_ndarray = np.vstack([vspace.A[np.newaxis, :] for vspace in sorted_vspaces[i:i+n]])
            # vspace_ndarray.shape = [vspace, vspace.n, vspace.dim]
            batch_subspace = subspace.A[np.newaxis, :].repeat(n, axis=0)

            X = np.matmul(
                    np.matmul(batch_subspace, vspace_ndarray.transpose((0, 2, 1))),
                    np.matmul(batch_subspace, vspace_ndarray.transpose((0, 2, 1))).transpose((0, 2, 1))
            )

            L, Q = np.linalg.eigh(X) # X is hermitian, can use eigh
            cossine_similarities.append(L)
            i += n

        cossine_similarities = np.vstack(cossine_similarities)

        # Unsort axis=0 cossine similarities. Sort axis=1 cossine_similarities by descending order
        z = zip(new_order, cossine_similarities)
        z = sorted(z)
        cossine_similarities = np.vstack([space for _, space in z])
        cossine_similarities = np.sort(cossine_similarities, axis=1)[:, ::-1]
        return cossine_similarities


# --- unittests
class TestMutualSubspaceMethod(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        pass
    
    def test_init(self):
        with self.assertRaises(TypeError):
            MutualSubspaceMethod()
        with self.assertRaises(ValueError):
            MutualSubspaceMethod(dim = -1)
        with self.assertRaises(ValueError):
            MutualSubspaceMethod(dim = 0)
        with self.assertRaises(ValueError):
            MutualSubspaceMethod(dim = "Hello")
        MutualSubspaceMethod(dim = self.dim)

    def test_cossine_similarity(self):
        msm = MutualSubspaceMethod(dim=3)
        subspace = VectorSpace(dim=3)
        subspace.append(np.array([[0, 0, 1], [0, 1, 0]]))
        vspace12 = VectorSpace(dim=3)
        vspace12.append(np.array([[1, 0, 0]]))
        vspace22 = VectorSpace(dim=3)
        vspace22.append(np.array([[1, 0, 0], [0, 0, 1]]))
        vspace32 = VectorSpace(dim=3)
        vspace32.append(np.array([[0, 1, 1], [1, 0, 0], [0, 0, -1]]))

        vspace_list = [vspace22, vspace32, vspace22, vspace32, vspace12, vspace12, vspace12]

        similarity = msm.cossine_similarity(vspace_list, subspace)
        self.assertTrue(np.allclose(similarity[4], np.zeros(2)))
        self.assertTrue(np.allclose(similarity[5], np.zeros(2)))
        self.assertTrue(np.allclose(similarity[6], np.zeros(2)))
        self.assertTrue(np.allclose(similarity[0], np.array([1.0, 0.0])))
        self.assertTrue(np.allclose(similarity[2], np.array([1.0, 0.0])))
    
    def test_subspace_similarities(self):
        msm = MutualSubspaceMethod(dim=3)
        subspace = VectorSpace(dim=3)
        subspace.append(np.array([[0, 0, 1], [0, 1, 0]]))
        vspace12 = VectorSpace(dim=3)
        vspace12.append(np.array([[1, 0, 0]]))
        vspace22 = VectorSpace(dim=3)
        vspace22.append(np.array([[1, 0, 0], [0, 0, 1]]))

        vspace_list = [vspace22, vspace12, vspace22, vspace12, vspace12, vspace22]

        self.assertTrue(np.allclose(msm.subspace_similarity(vspace_list, subspace), np.array([0.5000, 0.0000, 0.5000, 0.0000, 0.0000, 0.5000])))

    def test_train(self):
        msm = MutualSubspaceMethod(dim=32)
        mock_data = np.random.rand(100, 32)
        mock_labels = [i%10 for i in list(range(100))]
        msm.train(mock_data, mock_labels)
    
    def test_classify(self):
        msm = MutualSubspaceMethod(dim=32)
        mock_data = np.random.rand(100, 32)
        mock_labels = [i%10 for i in list(range(100))]
        msm.train(mock_data, mock_labels)

        mock_subspace1 = VectorSpace(dim=32)
        mock_subspace1.append(np.random.rand(6, 32))
        mock_subspace2 = VectorSpace(dim=32)
        mock_subspace2.append(np.random.rand(5, 32))
        mock_subspace3 = VectorSpace(dim=32)
        mock_subspace3.append(np.random.rand(7, 32))

        labels = msm.classify(mock_subspace1)
        self.assertEqual(len(labels), 1)

        labels = msm.classify([mock_subspace1, mock_subspace2, mock_subspace3])
        self.assertEqual(len(labels), 3)

    def test_eval(self):
        msm = MutualSubspaceMethod(dim=32)
        mock_data = np.random.rand(100, 32)
        mock_labels = [i%10 for i in list(range(100))]
        msm.train(mock_data, mock_labels)
        
        mock_subspaces = []
        for i in range(10):
            mock_subspaces.append(VectorSpace(dim=32))
            mock_subspaces[i].append(np.random.rand(6, 32))
        mock_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        eval = msm.eval(mock_subspaces, mock_labels)


if __name__ == "__main__":
    unittest.main()