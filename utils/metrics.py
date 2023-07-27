import numpy as np


def cal_tp_fp_fn(labels, preds):
    tp, fp, fn = 0, 0, 0

    for pred in preds:
        flag = 0
        for label in labels:
            if pred == label:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1
    fn = len(labels) - tp

    return np.array([tp, fp, fn])


def cal_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])
