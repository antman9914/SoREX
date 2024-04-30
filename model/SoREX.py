from typing import Union, Tuple
import logging, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor

class SoREX(torch.nn.Module):
    def __init__(self, in_channels, soc_mp_num, co_mp_num, K, num_total, num_user):
        super(SoREX, self).__init__()
        self.in_channels, self.num_total = in_channels, num_total
        self.soc_gcn = LGCN(self.in_channels, soc_mp_num)
        self.co_gcn = LGCN(self.in_channels, co_mp_num, option="sym")
        self.K = K
        self.num_user = num_user
        self.id_emb = Parameter(torch.nn.init.xavier_normal_(torch.empty((num_total, in_channels))))
        self.soc_id_emb = Parameter(torch.nn.init.xavier_normal_(torch.empty(num_total, in_channels)))

    def forward(self, inter_dict, soc_ext_weight, soc_ext_graph, co_graph, n_id, walks, split_range, soc_dict=None, neg_sample_num=10, r=0.7):
        test_flag = neg_sample_num == -1
        if test_flag:
            neg_sample_num = self.id_emb.size(0) - self.num_user
        device = n_id.device
        sqrt_dim = np.sqrt(self.in_channels)

        uid, pi_id = n_id[:, 0], n_id[:, 1]
        num_item = self.num_total - self.num_user
        soc_emb = self.soc_gcn(self.soc_id_emb, soc_ext_graph, soc_ext_weight)
        co_emb = self.co_gcn(self.id_emb, co_graph)

        co_emb = torch.cat([co_emb, torch.zeros((1, self.in_channels), device=device)], 0)
        soc_emb = torch.cat([soc_emb, torch.zeros((1, self.in_channels), device=device)], 0)
        id_emb = torch.cat([self.id_emb, torch.zeros((1, self.in_channels), device=device)], 0)
        soc_id_emb = torch.cat([self.soc_id_emb, torch.zeros((1, self.in_channels), device=device)], 0)

        neg_set = []
        true_pos_set = []
        
        cand_id_set = []
        all_anchor_set = []
        skip = []

        cand_soc_set = []
        soc_uid_mask = []

        for idx in range(uid.size(0)):
            cur_u = uid[idx]
            u_raw_neighbor = inter_dict[int(cur_u)] - self.num_user

            # Negative sampling
            if not test_flag:
                mask = torch.ones(num_item, dtype=torch.bool, device=device)
                mask[u_raw_neighbor] = False
                neg_idx_set = torch.arange(num_item, dtype=torch.int64, device=device)[mask]
                perm_idx = torch.randperm(neg_idx_set.size(0), device=device)
                perm_idx = perm_idx[:neg_sample_num]
                ni_id = neg_idx_set[perm_idx] + self.num_user
                neg_set.append(ni_id)
                # Remove the link for training
                u_raw_neighbor = u_raw_neighbor[u_raw_neighbor != pi_id[idx] - self.num_user]
                cand_id = torch.cat([pi_id[idx].unsqueeze(0), ni_id])
                cand_id_set.append(cand_id.unsqueeze(0))

                cur_skip = []
                for j in range(cand_id.size(0)):
                    cid = cand_id[j]
                    raw_neighbor = inter_dict[int(cid)]
                    if not test_flag and j == 0:
                        raw_neighbor = raw_neighbor[raw_neighbor != cur_u]
                    if raw_neighbor.size(0) == 0:
                        cur_skip.append(1)
                    else:
                        cur_skip.append(0)
                cur_skip = torch.tensor(cur_skip, dtype=torch.bool, device=device)
                skip.append(cur_skip.unsqueeze(0))

                if self.training and soc_dict is not None:
                    soc_raw_friend = soc_dict[int(cur_u)]
                    if len(soc_raw_friend) != 0:
                        pos_sample = torch.randperm(soc_raw_friend.size(0), device=device)[0]
                        mask = torch.ones(self.num_user, dtype=torch.bool, device=device)
                        mask[soc_raw_friend] = False
                        neg_idx_set = torch.arange(self.num_user, dtype=torch.int64, device=device)[mask]
                        perm_idx = torch.randperm(neg_idx_set.size(0), device=device)
                        perm_idx = perm_idx[:neg_sample_num]
                        ni_id = neg_idx_set[perm_idx]
                        soc_uid_mask.append(True)
                        cand_soc_set.append(torch.cat([pos_sample.unsqueeze(0), ni_id], -1))
                    else:
                        soc_uid_mask.append(False)

            true_pos_set.append(u_raw_neighbor)
        
        if self.training:
            cand_soc_set = torch.stack(cand_soc_set)
            soc_uid_mask = torch.tensor(soc_uid_mask, device=device, dtype=torch.bool)

        all_skip = []
        for idx in range(num_item):
            cid = int(idx + self.num_user)
            raw_neighbor = inter_dict[cid]
            all_anchor_set.append(raw_neighbor)
            if raw_neighbor.size(0) == 0:
                all_skip.append(1)
            else:
                all_skip.append(0)
        if self.training:
            for idx in range(uid.size(0)):
                cur_item_id = pi_id[idx] - self.num_user
                mask = all_anchor_set[cur_item_id] != uid[idx]
                all_anchor_set[cur_item_id] = all_anchor_set[cur_item_id][mask]
                if all_anchor_set[cur_item_id].size(0) == 0:
                    all_skip[cur_item_id] = 1
        all_skip = torch.tensor(all_skip, dtype=torch.bool, device=device)
        all_skip = torch.cat([torch.zeros(self.num_user, dtype=torch.bool, device=device), all_skip])

        if not test_flag:
            cand_id = torch.cat(cand_id_set, 0)
            skip = torch.cat(skip, 0)
        else:
            cand_id = torch.arange(num_item, dtype=torch.int64) + self.num_user
        cand_size = cand_id.size(-1)
        batch_size = uid.size(0)
        
        all_anchor_set = pad_sequence(all_anchor_set, batch_first=True, padding_value=-1)

        all_anchor_num = (all_anchor_set != -1).float().sum(-1, keepdim=True) + 1e-7
        all_anchor_soc = soc_emb[all_anchor_set].sum(1) / all_anchor_num
        
        u_ref_soc_emb = torch.cat([soc_emb[:self.num_user], all_anchor_soc, torch.zeros((1, self.in_channels), device=device)], 0)

        u_ref_co_emb_norm, u_ref_soc_emb_norm = F.normalize(co_emb, p=2, dim=-1), F.normalize(u_ref_soc_emb, p=2, dim=-1)
        
        patches = []
        all_co_sim, all_soc_sim = [], []
        hop_num = walks.size(1)
        for idx in range(batch_size):
            if test_flag:
                cur_cand = cand_id
            else:
                cur_cand = cand_id[idx]
            cur_walks = walks[split_range[idx]:split_range[idx+1]]
            cur_walks = cur_walks.view(-1)

            co_sim = (torch.mm(u_ref_co_emb_norm[cur_cand], u_ref_co_emb_norm[cur_walks].T) + 1) / 2
            soc_sim = (torch.mm(u_ref_soc_emb_norm[cur_cand], u_ref_soc_emb_norm[cur_walks].T) + 1) / 2

            co_sim = co_sim.view(cur_cand.size(0), -1, hop_num)
            soc_sim = soc_sim.view(cur_cand.size(0), -1, hop_num)
            cur_walks = cur_walks.view(-1, hop_num)

            co_sim4select = co_sim.mean(-1)
            soc_sim4select = soc_sim.mean(-1)
            
            co_selection_mask = torch.bernoulli(co_sim4select).bool()
            soc_selection_mask = torch.bernoulli(soc_sim4select).bool()
            co_selection_mask, soc_selection_mask = ~co_selection_mask, ~soc_selection_mask
            selected_walks = cur_walks.unsqueeze(0).repeat(cur_cand.size(0), 1, 1)

            cur_walks = cur_walks.view(-1)
            co_sim_new = torch.mm(u_ref_co_emb_norm[cur_cand], u_ref_co_emb_norm[cur_walks].T) / sqrt_dim
            soc_sim_new = torch.mm(u_ref_soc_emb_norm[cur_cand], u_ref_soc_emb_norm[cur_walks].T) / sqrt_dim
            co_sim_new, soc_sim_new = co_sim_new.view(cur_cand.size(0), -1, hop_num), soc_sim_new.view(cur_cand.size(0), -1, hop_num)
            co_sim_new = co_sim_new.masked_fill(co_selection_mask.unsqueeze(-1), -1e4)
            soc_sim_new = soc_sim_new.masked_fill(soc_selection_mask.unsqueeze(-1), -1e4)
            co_sim = co_sim.masked_fill(co_selection_mask.unsqueeze(-1), 0)
            soc_sim = soc_sim.masked_fill(soc_selection_mask.unsqueeze(-1), 0)
            all_co_sim.append(co_sim)
            all_soc_sim.append(soc_sim)
            co_sim, soc_sim = F.softmax(co_sim_new, dim=1), F.softmax(soc_sim_new, dim=1)

            co_patch = (id_emb[selected_walks] * co_sim.unsqueeze(-1)).sum(-2).sum(-2)
            soc_patch = (soc_id_emb[selected_walks] * soc_sim.unsqueeze(-1)).sum(-2).sum(-2)
            cur_patch = torch.cat([co_patch, soc_patch], -1)
            patches.append(cur_patch.unsqueeze(0))

        all_co_sim = torch.stack(all_co_sim)    # [B, cand_num, walk_num, hop_num]
        all_soc_sim = torch.stack(all_soc_sim)
        walks = walks.view(batch_size, -1, hop_num)     # [B, walk_num, hop_num]

        patches = torch.cat(patches, 0)
        if not test_flag:
            pos_patch = patches[:, 0]
            neg_patch = patches[:, 1:]
        else:
            pos_patch = []
            for idx in range(batch_size):
                pos_patch.append(patches[idx, pi_id[idx] - self.num_user].unsqueeze(0))
            pos_patch = torch.cat(pos_patch, 0)
            neg_patch = patches

        pos_patches, neg_patches = torch.split(pos_patch, self.in_channels, -1), torch.split(neg_patch, self.in_channels, -1)
        co_pos_patch, soc_pos_patch = pos_patches
        co_neg_patch, soc_neg_patch = neg_patches
        co_final_uemb = co_emb[uid] + co_pos_patch
        soc_final_uemb = soc_emb[uid] + soc_pos_patch
        co_final_uemb, soc_final_uemb = co_final_uemb / (hop_num + 1), soc_final_uemb / (hop_num + 1)

        co_pos_pred = (co_final_uemb * co_emb[pi_id]).sum(-1)
        soc_pos_pred = (soc_final_uemb * soc_id_emb[pi_id]).sum(-1)

        neg_co_uemb = co_emb[uid].unsqueeze(1) + co_neg_patch
        neg_soc_uemb = soc_emb[uid].unsqueeze(1) + soc_neg_patch
        neg_co_uemb, neg_soc_uemb = neg_co_uemb / (hop_num + 1), neg_soc_uemb / (hop_num + 1)

        if self.training:
            pred4soc = (soc_emb[uid[soc_uid_mask]].unsqueeze(1) * soc_emb[cand_soc_set]).sum(-1)
            pos_pred4soc, neg_pred4soc = pred4soc[:, 0], pred4soc[:, 1:].mean(1)
            ssl_loss = -F.logsigmoid(pos_pred4soc - neg_pred4soc).mean()

        if not test_flag:
            neg_set = torch.cat(neg_set).view(-1, neg_sample_num)
            co_neg_pred = (neg_co_uemb * co_emb[neg_set]).sum(-1)
            soc_neg_pred = (neg_soc_uemb * soc_id_emb[neg_set]).sum(-1)
        else:
            neg_idx = torch.arange(num_item) + self.num_user
            co_neg_pred = (neg_co_uemb * co_emb[neg_idx].unsqueeze(0)).sum(-1)
            soc_neg_pred = (neg_soc_uemb * soc_id_emb[neg_idx].unsqueeze(0)).sum(-1)

            masks = []
            for i in range(len(true_pos_set)):
                true_label = true_pos_set[i]
                mask = torch.zeros((1, num_item), dtype=torch.bool, device=device)
                mask[:, true_label] = True
                masks.append(mask)
            masks = torch.cat(masks, 0)
        if test_flag:
            co_neg_pred[masks] = -1e7
            soc_neg_pred[masks] = -1e7

        if self.training:
            return co_pos_pred, soc_pos_pred, co_neg_pred, soc_neg_pred, ssl_loss
        else:
            return co_pos_pred + soc_pos_pred, co_neg_pred + soc_neg_pred


class LGCN(torch.nn.Module):

    def __init__(self, in_channels, num_layers, option="mean"):
        super(LGCN, self).__init__()
        self.num_layers = num_layers
        self.in_channel = in_channels
        self.hidden_channel = in_channels
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if option == "mean":
                self.convs.append(SAGEConv(in_channels, linear=False))
            else:
                self.convs.append(LGCConv(in_channels))
    
    def forward(self, x, edge_index, edge_weight=None):
        
        final_x = x
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            final_x = final_x + x
        
        final_x = final_x / (self.num_layers + 1)
        return x


class Comp_LGCN(torch.nn.Module):
    def __init__(self, in_channels: int, num_layers: int):
        super(Comp_LGCN, self).__init__()
        self.num_layers = num_layers
        self.in_channel = in_channels
        self.hidden_channel = in_channels
        self.soc_convs = nn.ModuleList()
        self.co_convs = nn.ModuleList()
        for i in range(num_layers):
            self.soc_convs.append(SAGEConv(in_channels, linear=False))
            self.co_convs.append(SAGEConv(in_channels, linear=False))
        
    def forward(self, x, co_eindex, soc_eindex):
        final_x = x
        for i in range(self.num_layers):
            co_x = self.co_convs[i](x, co_eindex)
            soc_x = self.soc_convs[i](x, soc_eindex)
            x = (co_x + soc_x) / 2
            final_x = x + final_x
        x = final_x / (self.num_layers + 1)
        return x



class LGCConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 normalize: bool = False, **kwargs): 
        super(LGCConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.normalize = normalize
        self.improved = False
        self.add_self_loops = True

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        
        edge_index, edge_weight = gcn_norm(edge_index, None, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class SAGEConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 normalize: bool = False,
                 linear: bool = True, **kwargs):  
        super(SAGEConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.normalize = normalize
        self.linear = nn.Linear(in_channels, in_channels, bias=False) if linear else None
        self.reset_parameters()
    
    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        edge_weight = softmax(edge_weight, edge_index[1], None, x.size(0))
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        if self.linear is not None:
            out = self.linear(out)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
