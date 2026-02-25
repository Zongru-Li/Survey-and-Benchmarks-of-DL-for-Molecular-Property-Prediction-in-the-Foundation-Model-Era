import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.cluster import KMeans


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
        train_dataset = [dataset[i] for i in indices[:train_cutoff]]
        valid_dataset = [dataset[i] for i in indices[train_cutoff:valid_cutoff]]
        test_dataset = [dataset[i] for i in indices[valid_cutoff:]]
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
    
    def split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=None):
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)
        
        all_scaffolds = {}
        for i in range(N):
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


class ButinaSplitter(Splitter):
    """
    Butina clustering-based dataset splitter using Tanimoto similarity.
    
    This splitter uses the Butina clustering algorithm (Taylor-Butina clustering)
    to group similar molecules based on their Morgan fingerprints, then assigns
    entire clusters to train/valid/test sets to ensure chemical dissimilarity
    between splits.
    
    Reference: Butina, D. "Unsupervised Data Base Clustering Based on 
    Daylight's Fingerprint and Tanimoto Similarity: A Fast and Automated 
    Way To Cluster Small and Large Data Sets." J. Chem. Inf. Comput. Sci. 
    1999, 39, 4, 747–750.
    
    Parameters
    ----------
    cutoff : float, default 0.6
        Tanimoto similarity cutoff for clustering. Molecules with similarity
        above this threshold tend to be clustered together. Lower values
        produce smaller, tighter clusters; higher values produce larger,
        coarser clusters.
    fp_radius : int, default 2
        Radius for Morgan fingerprints.
    fp_bits : int, default 1024
        Number of bits for Morgan fingerprints.
    """
    
    def __init__(self, cutoff: float = 0.6, fp_radius: int = 2, fp_bits: int = 1024):
        super().__init__()
        self.cutoff = cutoff
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
    
    def _get_fingerprints(self, dataset):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        fps = []
        valid_indices = []
        for i, item in enumerate(dataset):
            smiles = item[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
                fps.append(fp)
                valid_indices.append(i)
        return fps, valid_indices
    
    def split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=None):
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        
        try:
            from rdkit import DataStructs
            from rdkit.ML.Cluster import Butina
        except ImportError:
            raise ImportError("RDKit is required for ButinaSplitter")
        
        fps, valid_indices = self._get_fingerprints(dataset)
        N = len(valid_indices)
        
        if N < 3:
            train_cutoff = int(frac_train * N)
            valid_cutoff = int((frac_train + frac_valid) * N)
            return (
                [dataset[valid_indices[i]] for i in range(train_cutoff)],
                [dataset[valid_indices[i]] for i in range(train_cutoff, valid_cutoff)],
                [dataset[valid_indices[i]] for i in range(valid_cutoff, N)]
            )
        
        dists = []
        nfps = len(fps)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])
        
        cluster_data = Butina.ClusterData(dists, nfps, self.cutoff, isDistData=True)
        cluster_sets = sorted(cluster_data, key=lambda x: -len(x))
        
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        
        train_idx = []
        valid_idx = []
        test_idx = []
        
        for cluster_set in cluster_sets:
            if len(train_idx) + len(cluster_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(cluster_set) > valid_cutoff:
                    test_idx.extend(cluster_set)
                else:
                    valid_idx.extend(cluster_set)
            else:
                train_idx.extend(cluster_set)
        
        train_dataset = [dataset[valid_indices[i]] for i in train_idx]
        valid_dataset = [dataset[valid_indices[i]] for i in valid_idx]
        test_dataset = [dataset[valid_indices[i]] for i in test_idx]
        
        return train_dataset, valid_dataset, test_dataset


class UMAPSplitter(Splitter):
    """
    UMAP-based dataset splitter following the paper's methodology.
    
    Reference: "Scaffold Splits Overestimate Virtual Screening Performance" (Guo et al., 2024)
    https://doi.org/10.26434/chemrxiv-2024-xj8r2
    
    Key methodology from the paper:
    - UMAP reduces dimensionality of molecular Morgan fingerprints
    - Clustering is performed on the reduced dimensions (paper uses 7 clusters for NCI-60)
    - Leave-one-cluster-out: entire clusters assigned to train/valid/test
    - This ensures molecules in different sets are truly dissimilar in chemical space
    
    UMAP Parameters (from paper and UMAP best practices for clustering):
    - n_neighbors: controls local vs global structure (default 15, paper context ~30 for large datasets)
    - min_dist: set to 0.0 for clustering (dense packing of points)
    - n_components: 2 (as used in paper's visualization and clustering)
    - metric: 'euclidean' (has gradient support, works well for fingerprints)
    """
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        metric: str = 'euclidean',
        n_clusters: int = 7,
        random_state: int = 42,
        fp_radius: int = 2,
        fp_bits: int = 2048
    ):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
    
    def _get_fingerprints(self, dataset):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        fps = []
        valid_indices = []
        for i, item in enumerate(dataset):
            smiles = item[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
                fps.append(np.array(fp))
                valid_indices.append(i)
        return np.array(fps), valid_indices
    
    def _compute_optimal_params(self, n_samples: int):
        n_neighbors = self.n_neighbors
        if n_samples < 50:
            n_neighbors = max(5, min(10, n_samples // 5))
        elif n_samples < 200:
            n_neighbors = max(10, min(15, n_samples // 10))
        elif n_samples < 1000:
            n_neighbors = 15
        elif n_samples < 10000:
            n_neighbors = 30
        else:
            n_neighbors = min(50, max(30, n_samples // 500))
        
        n_clusters = self.n_clusters
        if n_samples < 50:
            n_clusters = max(3, min(5, n_samples // 10))
        elif n_samples < 500:
            n_clusters = 5
        elif n_samples < 5000:
            n_clusters = 7
        else:
            n_clusters = min(12, max(7, n_samples // 1000))
        
        return n_neighbors, n_clusters
    
    def split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=None):
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        
        fps, valid_indices = self._get_fingerprints(dataset)
        N = len(valid_indices)
        
        if N < 10:
            rng = np.random.RandomState(seed or self.random_state)
            indices = list(range(N))
            rng.shuffle(indices)
            train_cutoff = int(frac_train * N)
            valid_cutoff = int((frac_train + frac_valid) * N)
            return (
                [dataset[valid_indices[i]] for i in indices[:train_cutoff]],
                [dataset[valid_indices[i]] for i in indices[train_cutoff:valid_cutoff]],
                [dataset[valid_indices[i]] for i in indices[valid_cutoff:]]
            )
        
        n_neighbors, n_clusters = self._compute_optimal_params(N)
        
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("umap-learn is required for UMAPSplitter. Install with: pip install umap-learn")
        
        random_state = seed if seed is not None else self.random_state
        
        reducer = UMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=random_state,
            n_jobs=1,
            low_memory=False if N < 10000 else True,
        )
        embedding = reducer.fit_transform(fps)
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init='auto'
        )
        cluster_labels = kmeans.fit_predict(embedding)
        
        cluster_to_indices = {}
        for cluster_id in range(n_clusters):
            cluster_to_indices[cluster_id] = np.where(cluster_labels == cluster_id)[0].tolist()
        
        clusters_with_size = [(cid, len(indices)) for cid, indices in cluster_to_indices.items()]
        clusters_with_size.sort(key=lambda x: -x[1])
        
        total_size = N
        n_train_target = int(frac_train * total_size)
        n_valid_target = int(frac_valid * total_size)
        n_test_target = total_size - n_train_target - n_valid_target
        
        train_clusters = []
        valid_clusters = []
        test_clusters = []
        
        train_count = 0
        valid_count = 0
        test_count = 0
        
        tolerance = 0.15
        
        for cluster_id, cluster_size in clusters_with_size:
            valid_ratio = valid_count / n_valid_target if n_valid_target > 0 else 1.0
            test_ratio = test_count / n_test_target if n_test_target > 0 else 1.0
            
            if test_ratio < (1.0 - tolerance):
                test_clusters.append(cluster_id)
                test_count += cluster_size
            elif valid_ratio < (1.0 - tolerance):
                valid_clusters.append(cluster_id)
                valid_count += cluster_size
            else:
                train_clusters.append(cluster_id)
                train_count += cluster_size
        
        train_idx = []
        valid_idx = []
        test_idx = []
        
        for cid in train_clusters:
            train_idx.extend(cluster_to_indices[cid])
        for cid in valid_clusters:
            valid_idx.extend(cluster_to_indices[cid])
        for cid in test_clusters:
            test_idx.extend(cluster_to_indices[cid])
        
        train_dataset = [dataset[valid_indices[i]] for i in train_idx]
        valid_dataset = [dataset[valid_indices[i]] for i in valid_idx]
        test_dataset = [dataset[valid_indices[i]] for i in test_idx]
        
        return train_dataset, valid_dataset, test_dataset
