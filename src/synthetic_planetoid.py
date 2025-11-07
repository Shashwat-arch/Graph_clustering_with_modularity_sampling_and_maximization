# src/synthetic_planetoid.py

import os
import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_scipy_sparse_matrix

def read_edge_list(path, num_nodes):
    # Read undirected edges, symmetrize, drop self-loops, deduplicate
    edges = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = int(parts[0]), int(parts[1])
            edges.append((a, b))
    if len(edges) == 0:
        # Empty graph safe-guard
        coo = sp.coo_matrix((num_nodes, num_nodes))
        return coo

    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)

    # Symmetrize like Planetoid citation graphs
    src = np.concatenate([src, dst], axis=0)
    dst = np.concatenate([dst, src[:len(dst)]], axis=0)

    data = np.ones_like(src, dtype=np.float32)
    coo = sp.coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    coo.sum_duplicates()
    coo.setdiag(0)
    coo.eliminate_zeros()
    return coo

def make_planetoid_masks(y, split='public',
                         num_train_per_class=20, num_val=500, num_test=1000,
                         seed=0):
    N = y.shape[0]
    rng = np.random.default_rng(seed)
    train = np.zeros(N, dtype=bool)
    val = np.zeros(N, dtype=bool)
    test = np.zeros(N, dtype=bool)

    classes = np.unique(y)
    # Per-class 20 labeled samples for train
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        take = min(num_train_per_class, len(idx_c))
        if take > 0:
            train[idx_c[:take]] = True

    remaining = np.where(~train)[0]
    rng.shuffle(remaining)
    nv = min(num_val, len(remaining))
    val[remaining[:nv]] = True

    remaining2 = remaining[nv:]
    nt = min(num_test, len(remaining2))
    test[remaining2[:nt]] = True

    return (
        torch.from_numpy(train),
        torch.from_numpy(val),
        torch.from_numpy(test),
    )

class SyntheticPlanetoid(InMemoryDataset):
    """
    Loads a single synthetic graph stored under:
      root/{name}/edge_list.txt
      root/{name}/features.npy
      root/{name}/labels.npy

    Produces a Planetoid-style Data with:
      x [N, F] float32, edge_index [2, E], y [N] int64,
      train_mask/val_mask/test_mask (bool).
    """
    def __init__(self, root, name, split='public',
                 num_train_per_class=20, num_val=500, num_test=1000,
                 seed=0, transform=None, pre_transform=None):
        self.name = name
        self.split = split
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        self.seed = seed
        super().__init__(root, transform, pre_transform)
        path = self.processed_paths[0]
        try:
            self.data, self.slices = torch.load(path, weights_only=False)
        except TypeError:
            # Older PyTorch versions do not accept weights_only kwarg
            self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        # Files live directly under {root}/{name}/...
        return [
            os.path.join(self.name, 'edge_list.txt'),
            os.path.join(self.name, 'features.npy'),
            os.path.join(self.name, 'labels.npy'),
        ]

    @property
    def processed_file_names(self):
        return f'{self.name}_planetoid.pt'

    def download(self):
        # No download: files are already present from the generator
        pass

    def process(self):
        base = os.path.join(self.root, self.name)
        edge_path = os.path.join(base, 'edge_list.txt')
        feat_path = os.path.join(base, 'features.npy')
        lab_path = os.path.join(base, 'labels.npy')

        if not os.path.exists(feat_path):
            raise FileNotFoundError(f'Missing features.npy at {feat_path}')
        if not os.path.exists(lab_path):
            raise FileNotFoundError(f'Missing labels.npy at {lab_path}')
        if not os.path.exists(edge_path):
            raise FileNotFoundError(f'Missing edge_list.txt at {edge_path}')

        x = np.load(feat_path)
        y = np.load(lab_path)
        if x.ndim != 2:
            raise ValueError(f'features.npy must be 2D [N, F], got shape {x.shape}')
        if y.ndim != 1:
            y = y.reshape(-1)
        N = x.shape[0]
        if len(y) != N:
            raise ValueError(f'Feature/label size mismatch: x has {N} rows, y has {len(y)} items')

        coo = read_edge_list(edge_path, N)
        edge_index, _ = from_scipy_sparse_matrix(coo)

        data = Data(
            x=torch.from_numpy(x).float(),
            edge_index=edge_index,
            y=torch.from_numpy(y).long()
        )
        train_mask, val_mask, test_mask = make_planetoid_masks(
            data.y.numpy(), split=self.split,
            num_train_per_class=self.num_train_per_class,
            num_val=self.num_val, num_test=self.num_test, seed=self.seed
        )
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        data_list = [data]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        torch.save(self.collate(data_list), self.processed_paths[0])
