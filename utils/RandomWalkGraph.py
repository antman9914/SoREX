from numba import njit
from scipy.sparse import csr_matrix
import numpy as np

@njit(nogil=True)
def _csr_row_cumsum(indptr, data):
    out = np.empty_like(data)
    for i in range(len(indptr) - 1):
        acc = 0
        for j in range(indptr[i], indptr[i + 1]):
            acc += data[j]
            out[j] = acc
        out[j] = 1.0
    return out


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]

@njit(nogil=True)
def _random_walk(indptr, indices, walk_length, t, tar, bans):

    walk = np.full(walk_length, -1, dtype=indices.dtype)
    walk[0] = t
    raw_neighbors = _neighbors(indptr, indices, t)
    if len(bans) != 0:
        raw_neighbors = raw_neighbors[raw_neighbors != bans[0]]
    if raw_neighbors.shape[0] == 0:
        return walk
    walk[1] = np.random.choice(raw_neighbors)
    bans = [walk[0]]
    for j in range(2, walk_length):
        if indptr[walk[j - 1]] >= indptr[walk[j - 1] + 1]:
            break
        if walk[j-1] == tar:
            walk[j] = walk[j-1]
            continue
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        mask = np.ones_like(neighbors, dtype=np.bool_)
        for blocker in bans:
            mask = (neighbors != blocker) & mask
        neighbors = neighbors[mask]
        bans.append(walk[j-1])
        if neighbors.shape[0] == 0:
            break
        walk[j] = np.random.choice(neighbors)
    return walk


@njit(nogil=True)
def _random_walk_weighted(indptr, indices, data, walk_length, t, tar, bans):

    walk = np.full(walk_length, -1, dtype=indices.dtype)
    walk[0] = t
    raw_neighbors = _neighbors(indptr, indices, t)
    raw_data = _neighbors(indptr, data, t)
    if len(bans) != 0:
        mask = raw_neighbors != bans[0]
        raw_neighbors_filter = raw_neighbors[mask]
    if raw_neighbors_filter.shape[0] == 0:
        return walk
    walk[1] = raw_neighbors[np.searchsorted(raw_data, np.random.rand())]
    while walk[1] == bans[0]:
        walk[1] = raw_neighbors[np.searchsorted(raw_data, np.random.rand())]
    bans = [walk[0]]
    for j in range(2, walk_length):
        if indptr[walk[j - 1]] >= indptr[walk[j - 1] + 1]:
            break
        if walk[j-1] == tar:
            walk[j] = walk[j-1]
            continue
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        neighbors_p = _neighbors(indptr, data, walk[j - 1])
        mask = np.ones_like(neighbors, dtype=np.bool_)
        for blocker in bans:
            mask = (neighbors != blocker) & mask
        neighbors_filter = neighbors[mask]
        if neighbors_filter.shape[0] == 0:
            break
        walk[j] = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
        while walk[j] in bans:
            walk[j] = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
        bans.append(walk[j-1])
    return walk


class RandomWalkGraph:
    def __init__(self, num_nodes: int, src: np.ndarray, dst: np.ndarray, data: np.ndarray =None):
        self.num_nodes = num_nodes
        if data is None:
            self.is_weighted = False
            data = np.ones(len(src), dtype=bool)
        else:
            self.is_weighted = True
        
        edges = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
        # edges.sort_indices()
        self.indptr = edges.indptr
        self.indices = edges.indices
        if self.is_weighted:
            data = edges.data / edges.sum(axis=1).A1.repeat(np.diff(self.indptr))
            self.data = _csr_row_cumsum(self.indptr, data)

    def generate_random_walk(self, walk_length, start, tar, block=None):
        if block is None:
            block = np.empty(0, dtype=np.int64)
        if self.is_weighted:
            walk = _random_walk_weighted(
                self.indptr, self.indices, self.data, walk_length, start, tar, block
            )
        else:
            walk = _random_walk(self.indptr, self.indices, walk_length, start, tar, block)
        return walk
