import torch
import numpy as np
from torch.utils.data import DataLoader


class SRLoader(DataLoader):
    def __init__(self, edge_index, graph4walk, num_walk, num_hop, mode, **kwargs):
        self.edge_index = edge_index.t()
        edge_size = edge_index.size(1) // 2
        self.dataset = torch.arange(edge_size).reshape(-1, 1)
        self.walkgraph = graph4walk
        self.num_walk = num_walk
        self.num_hop = num_hop
        self.mode = mode
        super(SRLoader, self).__init__(self.dataset, collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        batch = torch.cat(batch)
        pos_samples = self.edge_index[batch]

        nids = pos_samples[:, 0].numpy()
        blockers = pos_samples[:, 1].numpy()
        num = 0
        selected_walks = []
        split_range = [0]
        while num < self.num_walk:
            num += 1
            for idx in range(nids.shape[0]):  
                nid = nids[idx]   
                if self.mode != 'train':
                    walk_sample = self.walkgraph.generate_random_walk(self.num_hop, start=nid, tar=blockers[idx])
                else:
                    bans = [blockers[idx]]
                    walk_sample = self.walkgraph.generate_random_walk(self.num_hop, start=nid, tar=blockers[idx], block=np.array(bans, dtype=np.int64))
                selected_walks.append(np.expand_dims(walk_sample, 0))
        selected_walks = np.concatenate(selected_walks, 0).reshape(self.num_walk, -1, self.num_hop).transpose((1, 0, 2))
        filtered_walks = []
        for i in range(nids.shape[0]):
            unique_walks = selected_walks[i]
            filtered_walks.append(unique_walks)
            split_range.append(split_range[-1] + unique_walks.shape[0])
        selected_walks = np.concatenate(filtered_walks, 0)
        return (pos_samples, torch.tensor(selected_walks[:, 1:], dtype=torch.int64), split_range)

