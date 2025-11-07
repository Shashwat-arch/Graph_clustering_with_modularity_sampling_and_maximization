# synthetic_planetoid.py
import os
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp

def read_edge_list(path, num_nodes):
    # Supports undirected edges; deduplicate and build COO
    edges = []
    with open(path, 'r') as f:
        for line in f:
            a, b = line.strip().split()
            edges.append((int(a), int(b)))
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    # Make symmetric like Planetoid citation graphs
    src = np.concatenate([src, dst], axis=0)
    dst = np.concatenate([dst, src[:len(dst)]], axis=0)
    coo = sp.coo_matrix((np.ones_like(src), (src, dst)), shape=(num_nodes, num_nodes))
    coo.sum_duplicates()
    coo.setdiag(0)
    coo.eliminate_zeros()
    return coo

def make_planetoid_masks(y, split='public', num_train_per_class=20, num_val=500, num_test=1000, seed=0):
    N = y.shape[0]
    rng = np.random.default_rng(seed)
    train = np.zeros(N, dtype=bool)
    val = np.zeros(N, dtype=bool)
    test = np.zeros(N, dtype=bool)
    classes = np.unique(y)
    if split in ('public', 'random', 'full'):
        # Planetoid-style: 20 per class for train, 500 val, 1000 test where possible
        idx_all = np.arange(N)
        # Per-class train
        for c in classes:
            idx_c = idx_all[y == c]
            rng.shuffle(idx_c)
            take = min(num_train_per_class, len(idx_c))
            train[idx_c[:take]] = True
        remaining = np.where(~train)[0]
        rng.shuffle(remaining)
        nv = min(num_val, len(remaining))
        val[remaining[:nv]] = True
        remaining2 = remaining[nv:]
        nt = min(num_test, len(remaining2))
        test[remaining2[:nt]] = True
        # If graph smaller than 20/500/1000 totals, masks will clip automatically
    else:
        raise ValueError('Unsupported split')
    return torch.from_numpy(train), torch.from_numpy(val), torch.from_numpy(test)

class SyntheticPlanetoid(InMemoryDataset):
    def __init__(self, root, name, split='public',
                 num_train_per_class=20, num_val=500, num_test=1000,
                 transform=None, pre_transform=None, force_reload=False, seed=0):
        self.name = name
        self.split = split
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        self.seed = seed
        super().__init__(root, transform, pre_transform)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        # Expect the generator’s files
        return [
            os.path.join(self.name, 'edge_list.txt'),
            os.path.join(self.name, 'features.npy'),
            os.path.join(self.name, 'labels.npy'),
        ]

    @property
    def processed_file_names(self):
        return f'{self.name}_planetoid.pt'

    def download(self):
        # Not used; files already created by your generator
        pass

    def process(self):
        base = os.path.join(self.raw_dir, self.name)
        x = np.load(os.path.join(base, 'features.npy'))
        y = np.load(os.path.join(base, 'labels.npy'))
        N = x.shape[0]
        # Prefer adjacency from edge list to mirror Planetoid style
        coo = read_edge_list(os.path.join(base, 'edge_list.txt'), N)
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
