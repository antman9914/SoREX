import numpy as np

def ndcg_score(y_true, y_score, k=10):
    y_standard = np.zeros_like(y_true)
    for i in range(y_standard.shape[1]):
        y_standard[:, i] = i
    best = dcg_score(y_true, y_standard, k)
    actual = dcg_score(y_true, y_score, k)
    result = actual / best
    result[result is np.nan] = 0
    return result


def dcg_score(y_true, y_pred, k=10):
    y_true = np.take(y_true, y_pred[:, :k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(np.shape(y_true)[1]) + 2)
    return np.sum(gains / discounts, axis=1)


def hit_rate(logits, label, k=10):
    sorted_idx = logits[:, :k]
    size = logits.shape[0]
    tot = 0.
    for i in range(size):
        if label[i] in sorted_idx[i, :]:
            tot += 1
    return tot


def top_k_acc(sorted_idx, label, k=10):
    size = sorted_idx.shape[0]
    tot = 0.
    for i in range(size):
        # intersect = np.intersect1d(label[i], sorted_idx[i, :k])
        cur_tot = 0
        for ele in label[i]:
            if ele in sorted_idx[i, :k]:
                cur_tot += 1
        tot += cur_tot / len(label[i])
        # if len(intersect) > 0:
        #     tot += 1
    return tot