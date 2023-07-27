import sys

import torch
import numpy as np
import logging
from utils.metrics import cal_tp_fp_fn, cal_p_r_f
from utils.decode import decode


logger = logging.getLogger(__name__)


def cal_entity_proportion(labels, entity_types):
    entity_proportion = {}
    entity_num = 0
    for entity_type in entity_types:
        entity_proportion.update({entity_type: 0})
    for label in labels:
        for key in label:
            entity_proportion[key] += len(label[key])
            entity_num += len(label[key])
    for key in entity_proportion:
        entity_proportion[key] /= entity_num
    return entity_proportion


def count_entity_type(ent2id):
    entity_types = []
    for key in ent2id:
        if '-' in key:
            entity_type = key.split('-')[1]
            if entity_type not in entity_types:
                entity_types.append(entity_type)
    return entity_types


def evaluate(model, device, data_loader, ent2id):
    labels = []
    preds = []

    model.eval()
    with torch.no_grad():
        for index, batch_data in enumerate(data_loader):
            for key in batch_data.keys():
                if key == 'labels':
                    for label in batch_data[key]:
                        labels.append(label)
                    continue
                batch_data[key] = batch_data[key].to(device)
            batch_preds = model(mode='evaluate', **batch_data)[0]
            for pred in batch_preds:
                preds.append(pred)

    entity_types = count_entity_type(ent2id)
    entity_types_num = len(entity_types)
    entities_labels = []
    entities_preds = []
    for label, pred in zip(labels, preds):
        if type(pred) == torch.Tensor:
            pred = pred.numpy().tolist()
        if type(label) == torch.Tensor:
            label = label[0: len(pred)].numpy().tolist()
        assert len(label) == len(pred), 'lengths of label({}) and pred({}) are not equal'.format(len(label), len(pred))
        entities_labels.append(decode(label, ent2id))
        entities_preds.append(decode(pred, ent2id))

    entity_proportion = cal_entity_proportion(entities_labels, entity_types)

    tp_fp_fn = np.zeros([entity_types_num, 3])
    p_r_f = np.zeros(3)

    for label, pred in zip(entities_labels, entities_preds):
        single_tp_fp_fn = np.zeros([entity_types_num, 3])
        for index, entity_type in enumerate(entity_types):
            single_tp_fp_fn[index] += cal_tp_fp_fn(label[entity_type], pred[entity_type])
        tp_fp_fn += single_tp_fp_fn

    for index, entity_type in enumerate(entity_types):
        type_p_r_f = cal_p_r_f(tp_fp_fn[index][0], tp_fp_fn[index][1], tp_fp_fn[index][2])
        p_r_f += type_p_r_f * entity_proportion[entity_type]

    logger.info(f'[MIRCO] precision: {p_r_f[0]:.4f}, recall: {p_r_f[1]:.4f}, f1: {p_r_f[2]:.4f}')

    return p_r_f
