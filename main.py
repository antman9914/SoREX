import torch
import numpy as np
import random, time, json, os
import argparse
import logging
import pickle
import torch
import torch.nn.functional as F
from torch.cuda import amp
from utils.dataloader import SRLoader
from utils.RandomWalkGraph import RandomWalkGraph
from torch_geometric.utils import softmax
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from model.SoREX import SoREX
from utils.metric import *

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
# torch.autograd.set_detect_anomaly(True)
# faulthandler.enable()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='running mode, choose between [train, test]')
parser.add_argument('--dataset', type=str, default='yelp', help='dataset chosen for training')
parser.add_argument('--note', type=str, default='init', help='model under verification in this run')
parser.add_argument('--checkpoint_dir', '-cd', type=str, default='checkpoint', help='path of checkpoint file')
parser.add_argument('--log_dir', '-ld', type=str, default="log", help="path of stored logs")
parser.add_argument('--gpu_id', type=int, default=-1, help='gpu id chosen to run train/test')
parser.add_argument('--soc_mp_layer', type=int, default=2, help="Message passing hops on social graph")
parser.add_argument('--co_mp_layer', type=int, default=1, help="Message passing hops on co-purchase graphs")
parser.add_argument('--K', type=int, default=10, help="The number of selected most relevant neighbors")
parser.add_argument('--num_walk', type=int, default=100, help="The number of sampled random walks")
parser.add_argument('--num_hop', type=int, default=2, help="The length of sampled random walks")
parser.add_argument('--input_dim', type=int, default=64, help='dimension of input feature')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='hyperparameter of weight decay')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--test_bs', type=int, default=64, help='batch size for test and validation')
parser.add_argument('--neg_sample_num', type=int, default=8, help='number of negative samples during training/validation')
parser.add_argument('--eval_per_n', type=int, default=1000, help='evaluate per n steps')
parser.add_argument('--epoch_num', type=int, default=30, help='training epoch number')
parser.add_argument('--early_stop', type=int, default=5, help="Early stop duration")

args, _ = parser.parse_known_args()
if args.mode == 'train':
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    os.makedirs('%s/%s' % (args.checkpoint_dir, time_str))
    os.makedirs('%s/%s' % (args.log_dir, time_str))
    log_time_str = time_str
else:
    time_str = args.time_str
    log_time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    os.makedirs('%s/%s' % (args.log_dir, log_time_str))
checkpoint_path = '%s/%s/%s.pth' % (args.checkpoint_dir, time_str, '_'.join([args.note, args.dataset]))
logging.basicConfig(filename="%s/%s/log_%s.txt" % (args.log_dir, log_time_str, args.note), format = '%(asctime)s - %(message)s', level=logging.INFO, filemode='a')

logging.info("Run %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
logging.info("Current process ID: %d" % os.getpid())
logging.info("GPU ID: %d" % args.gpu_id)
logging.info("Current Dataset: %s" % args.dataset)
logging.info("Running Mode: %s" % args.mode)
logging.info("------HyperParameter Settings------")
logging.info("Social Diffusion Layers: %d" % args.soc_mp_layer)
logging.info("Co-Purchase Diffusion Layers: %d" % args.co_mp_layer)
logging.info("Number of selected neighbors: %d" % args.K)
logging.info("Number of sampled walks: %d" % args.num_walk)
logging.info("Length of sampled walks: %d" % args.num_hop)
logging.info("ID Embedding Dimension: %d" % args.input_dim)
if args.mode == 'train':
    logging.info("Learning Rate: %f" % args.lr)
    logging.info("Weight Decay: %f" % args.weight_decay)
    logging.info("Batch Size: %d" % args.batch_size)
    logging.info("Negative Sample: %d" % args.neg_sample_num)
    logging.info("Epoch Number: %d" % args.epoch_num)
    logging.info("Evaluate per %d step" % args.eval_per_n)
    logging.info("\n\n")
else:
    logging.info("Batch Size: %d" % args.test_bs)
    logging.info("\n\n")

if args.gpu_id != -1:
    device = torch.device('cuda:%d' % args.gpu_id if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
enable_amp = True if "cuda" in device.type else False

# Data loading
logging.info("Loading data...")
inter_graph, val_patch, test_patch, soc_graph, train_impact, test_impact = pickle.load(open('data/' + args.dataset + "/" + args.dataset + '.pkl', 'rb'))
inter_graph, val_patch, test_patch, soc_graph = torch.tensor(inter_graph, dtype=torch.int64), torch.tensor(val_patch, dtype=torch.int64), torch.tensor(test_patch, dtype=torch.int64), torch.tensor(soc_graph, dtype=torch.int64)
test_graph = torch.cat([inter_graph, val_patch], dim=-1)
train_impact, test_impact = torch.sqrt(train_impact).to(device), torch.sqrt(test_impact).to(device)
values_one = torch.ones(inter_graph.size(1))
values_one_test = torch.ones(test_graph.size(1))
num_total, num_user = int(torch.max(inter_graph[0])) + 1, int(torch.max(soc_graph[0])) + 1
logging.info("User number: %d; Item number: %d" % (num_user, num_total - num_user))
inter_graph = torch.sparse_coo_tensor(inter_graph, values_one, (num_total, num_total)).to(device).coalesce()
test_graph = torch.sparse_coo_tensor(test_graph, values_one_test, (num_total, num_total)).to(device).coalesce()
soc_adj = torch.sparse_coo_tensor(soc_graph, torch.ones(soc_graph.size(1)), (num_user, num_user)).coalesce().to(device)
soc_ext = torch.arange(num_user, num_total, dtype=torch.int64, device=device).unsqueeze(0)
train_impact_ext = torch.cat([train_impact, torch.zeros(soc_ext.size(1), device=device)], -1)
test_impact_ext = torch.cat([test_impact, torch.zeros(soc_ext.size(1), device=device)], -1)
soc_ext = torch.cat([soc_ext, soc_ext], 0)
soc_ext = torch.cat([soc_graph.to(device), soc_ext], -1)
soc_ext_adj = torch.sparse_coo_tensor(soc_ext, torch.ones(soc_ext.size(1), device=device), (num_total, num_total)).coalesce()
soc_graph = soc_graph.to(device)

inter_dict, test_dict = {}, {}
soc_dict, test_soc_dict = {}, {}
max_context_cnt = 30
logging.info("Transforming Graphs...")
for i in range(num_total):
    inter_dict[i] = inter_graph[i]._indices()[0]
    test_dict[i] = test_graph[i]._indices()[0]
for i in range(num_total):
    if i < num_user:
        soc_dict[i] = soc_adj[i]._indices()[0]
        test_soc_dict[i] = soc_adj[i]._indices()[0]
inter_ext_eindex, inter_eweight = gcn_norm(inter_graph.indices(), None, num_total)
test_ext_eindex, test_eweight = gcn_norm(test_graph.indices(), None, num_total)
self_loop = torch.sparse_coo_tensor((range(num_user), range(num_user)), [1.] * num_user).to(device)
soc_graph_w_loop = soc_adj + self_loop
soc_deg_vec = torch.sparse.sum(soc_graph_w_loop, dim=0).values() - 1
self_loop = torch.sparse_coo_tensor((range(num_total), range(num_total)), [1.] * num_total).to(device)
inter_graph_w_loop = inter_graph + self_loop
test_graph_w_loop = test_graph + self_loop
inter_deg_vec = torch.sparse.sum(inter_graph_w_loop, dim=0).values()[:num_user] - 1
test_deg_vec = torch.sparse.sum(test_graph_w_loop, dim=0).values()[:num_user] - 1
overall_deg_vec = (test_deg_vec + soc_deg_vec).cpu().numpy()
train_ratio = soc_deg_vec / (inter_deg_vec + 1e-7)
train_ratio[inter_deg_vec == 0] = 1
test_ratio = soc_deg_vec / (test_deg_vec + 1e-7)
test_ratio[test_deg_vec == 0] = 1
soc_eweight = softmax(train_impact, soc_graph[1], None, num_total)
soc_eweight = soc_eweight * train_ratio[soc_graph[0]]
test_soc_eweight = softmax(test_impact, soc_graph[1], None, num_total)
test_soc_eweight = test_soc_eweight * test_ratio[soc_graph[0]]
inter_mask, test_mask = (inter_ext_eindex[0] != inter_ext_eindex[1]), (test_ext_eindex[0] != test_ext_eindex[1])
inter_ext_eindex, test_ext_eindex = inter_ext_eindex[:, inter_mask], test_ext_eindex[:, test_mask]
inter_eweight, test_eweight = inter_eweight[inter_mask], test_eweight[test_mask]
inter_eweight = softmax(inter_eweight, inter_ext_eindex[1], None, num_total)
test_eweight = softmax(test_eweight, test_ext_eindex[1], None, num_total)
full_eindex = torch.cat([inter_ext_eindex, soc_graph], -1)
full_eindex_test = torch.cat([test_ext_eindex, soc_graph], -1)
full_eweigh, full_eweigh_test = torch.cat([inter_eweight, soc_eweight]), torch.cat([test_eweight, test_soc_eweight])
full_eindex, full_eindex_test = full_eindex.cpu().numpy(), full_eindex_test.cpu().numpy()
val_graph4walk = RandomWalkGraph(num_total, full_eindex[0], full_eindex[1])
test_graph4walk = RandomWalkGraph(num_total, full_eindex_test[0], full_eindex_test[1])

t_loader = SRLoader(inter_graph.indices().cpu(), val_graph4walk, args.num_walk, args.num_hop, "train", batch_size=args.batch_size, shuffle=True)
v_loader = SRLoader(val_patch, val_graph4walk, args.num_walk, args.num_hop, "test", batch_size=args.batch_size, shuffle=False)
test_loader = SRLoader(test_patch, test_graph4walk, args.num_walk, args.num_hop, "test", batch_size=args.test_bs, shuffle=False)

logging.info("Training set size: %d; Val set size: %d; Test set size: %d" % (t_loader.dataset.size(0), v_loader.dataset.size(0), test_loader.dataset.size(0)))

model = SoREX(args.input_dim, args.soc_mp_layer, args.co_mp_layer, args.K, num_total, num_user)
model = model.to(device)
if args.mode == 'train':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler(enabled=enable_amp)

def train(best_ndcg):
    model.train()
    step = 0
    for batch in t_iter:
        optimizer.zero_grad()
        batch_sample, selected_walks, split_range = batch
        batch = batch_sample.to(device)
        selected_walks = selected_walks.to(device)
        
        with amp.autocast(enabled=enable_amp):
            pos_logit_1, pos_logit_2, neg_logit_1, neg_logit_2, ss_loss = model(inter_dict, train_impact_ext, soc_ext, inter_graph.indices(), batch, selected_walks, split_range, neg_sample_num=args.neg_sample_num, soc_dict=soc_dict)
            neg_logit_1, neg_logit_2 = neg_logit_1.mean(1), neg_logit_2.mean(1)
            loss = -F.logsigmoid(pos_logit_1 - neg_logit_1).mean() - F.logsigmoid(pos_logit_1 + pos_logit_2 - neg_logit_1 - neg_logit_2).mean() - F.logsigmoid(pos_logit_2 - neg_logit_2).mean()
            loss = loss + ss_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step % 30 == 0:
            logging.info(loss)
        step += 1
    start = time.time()
    v_iter = iter(v_loader)
    hr_10, mrr, ndcg, _, _ = test('val', v_iter)
    end = time.time()
    logging.info("time consumption: %.6f" % (end-start))
    logging.info("Test result on validation set: HR@10 %.6f, MRR: %.6f, NDCG@10: %.6f" % (hr_10, mrr, ndcg))
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        logging.info("New model saved!")
        torch.save(model.state_dict(), checkpoint_path)
    return best_ndcg


@torch.no_grad()
def test(mode, v_iter):
    model.eval()
    logging.info("Start test...")
    if mode == 'test':
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    step = 0
    hr_10_tot, mrr_tot, ndcg_tot = [], [], []
    sample_num = []
    top_k_rank_list = []
    for batch in v_iter:
        batch_sample, selected_walks, split_range = batch
        batch = batch_sample.to(device)
        selected_walks = selected_walks.to(device)
        with amp.autocast(enabled=enable_amp):
            if mode == 'val':
                pos_logit, neg_logit = model(inter_dict, train_impact_ext, soc_ext, inter_graph.indices(), batch, selected_walks, split_range, neg_sample_num=1000)
            else:
                pos_logit, neg_logit = model(test_dict, test_impact_ext, soc_ext, test_graph.indices(), batch, selected_walks, split_range, neg_sample_num=-1)
            pos_logit = pos_logit.unsqueeze(-1)
            logits = torch.cat([pos_logit, neg_logit], 1)
            sample_num.append(logits.size(0))
            sorted_idx = torch.argsort(logits, dim=-1, descending=True)
            top_k_list = sorted_idx[:, :10] - 1
            for idx in range(sample_num[-1]):
                mask = top_k_list[idx] == -1
                top_k_list[idx, mask] = batch[idx, 1]
            top_k_rank_list.append(top_k_list)
            sorted_idx = sorted_idx.cpu().numpy()
        hit_idx = []
        for i in range(sorted_idx.shape[0]):
            if 0 in sorted_idx[i, :5]:
                hit_idx.append(i)
        hit_idx = np.array(hit_idx, dtype=np.int64)

        if step % 50 == 0:
            logging.info(step)
        label = np.zeros(batch.size(0), dtype=np.int64)
        hr_10 = hit_rate(sorted_idx, label, k=10) / logits.size(0)
        hr_10_tot.append(hr_10)
        y_true = np.zeros_like(sorted_idx)
        for i in range(sorted_idx.shape[0]):
            y_true[i, label[i]] = 1
            y_true[i] = y_true[i][sorted_idx[i]]
        rr_score = y_true / (np.arange(np.shape(y_true)[1]) + 1)
        mrr = np.sum(rr_score) / np.sum(y_true)
        mrr_tot.append(mrr)

        y_true = np.zeros_like(sorted_idx)
        y_true[:, 0] = 1
        ndcg = ndcg_score(y_true, sorted_idx, k=10)
        ndcg_tot.append(np.mean(ndcg))
        step += 1
    sample_num = np.array(sample_num)
    top_k_rank_list = torch.cat(top_k_rank_list, 0)
    cover_item = torch.unique(top_k_rank_list).cpu()
    top_k_rank_list = top_k_rank_list.long().view(-1)
    freqs = torch.bincount(top_k_rank_list)
    freqs = freqs / freqs.sum()
    entropy = - (freqs * torch.log(freqs + 1e-7)).sum()

    return np.sum(hr_10_tot * sample_num) / np.sum(sample_num),\
            np.sum(mrr_tot * sample_num) / np.sum(sample_num),\
            np.sum(ndcg_tot * sample_num) / np.sum(sample_num), \
            cover_item.size(0) / (num_total - num_user), entropy


best_ndcg = 0
if args.mode == 'train':
    logging.info("Start training...")
    best_necg = 0
    duration = 0
    for epoch in range(1, args.epoch_num + 1):
        t_iter = iter(t_loader)
        logging.info("Current epoch %d:" % epoch)
        pre_best = best_ndcg
        best_ndcg = train(best_ndcg)
        if best_ndcg == pre_best:
            duration += 1
            if duration == args.early_stop:
                logging.info("Converged. Stop Training.")
                break
        else:
            duration = 0
start = time.time()
v_iter = iter(test_loader)
hr_10, mrr, ndcg_10, coverage, entropy = test('test', v_iter)
end = time.time()
logging.info("time consumption: %.6f" % (end-start))
logging.info("Test result on test set: HR@10 %.6f, MRR: %.6f, NDCG@10: %.6f" % (hr_10, mrr, ndcg_10))
logging.info("Diversity measure: Coverage %.6f, Entropy %.6f" % (coverage, entropy))
