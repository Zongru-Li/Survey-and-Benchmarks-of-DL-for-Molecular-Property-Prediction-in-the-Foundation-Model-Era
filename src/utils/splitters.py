import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold


def generate_scaffold(smiles, include_chirality=False):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


class Splitter(object):
    def __init__(self):
        super(Splitter, self).__init__()


class RandomSplitter(Splitter):
    def __init__(self):
        super(RandomSplitter, self).__init__()

    def split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=None):
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)
        indices = list(range(N))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)
        train_dataset = dataset[indices[:train_cutoff]]
        valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
        test_dataset = dataset[indices[valid_cutoff:]]
        return train_dataset, valid_dataset, test_dataset


class IndexSplitter(Splitter):
    def __init__(self):
        super(IndexSplitter, self).__init__()

    def split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)
        indices = list(range(N))
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)
        train_dataset = dataset[indices[:train_cutoff]]
        valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
        test_dataset = dataset[indices[valid_cutoff:]]
        return train_dataset, valid_dataset, test_dataset


class ScaffoldSplitter(Splitter):
    def __init__(self):
        super(ScaffoldSplitter, self).__init__()
    
    def split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)
        
        all_scaffolds = {}
        for i in range(N):
            smiles = dataset[i][0]
            scaffold = generate_scaffold(dataset[i][0], include_chirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)
        
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        train_dataset = [dataset[i] for i in train_idx]
        valid_dataset = [dataset[i] for i in valid_idx]
        test_dataset = [dataset[i] for i in test_idx]

        return train_dataset, valid_dataset, test_dataset


class RandomScaffoldSplitter(Splitter):
    def __init__(self):
        super(RandomScaffoldSplitter, self).__init__()
    
    def split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=None):
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)
        rng = np.random.RandomState(seed)

        scaffolds = defaultdict(list)
        for ind in range(N):
            scaffold = generate_scaffold(dataset[ind]['smiles'], include_chirality=True)
            scaffolds[scaffold].append(ind)

        scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

        n_total_valid = int(np.floor(frac_valid * len(dataset)))
        n_total_test = int(np.floor(frac_test * len(dataset)))

        train_idx = []
        valid_idx = []
        test_idx = []

        for scaffold_set in scaffold_sets:
            if len(valid_idx) + len(scaffold_set) <= n_total_valid:
                valid_idx.extend(scaffold_set)
            elif len(test_idx) + len(scaffold_set) <= n_total_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset
