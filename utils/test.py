from utils.load_model_and_parallel import load_model_and_parallel
import argparse
import json
import os
from utils.processor import raw_data_to_dataset
from torch.utils.data import DataLoader, RandomSampler
from utils.evaluate import evaluate
from utils.predict import predict
import logging


def test(model_path, model_type, gpu_ids, data_dir, ent2id, bert_dir, label_type_num, strict=True, ent2id_way=None,
         batch_size=64, mode='evaluate'):
    if type(ent2id) == str:
        assert ent2id_way is not None
        ent2id = json.load(open(os.path.join(ent2id, ent2id_way + '_ent2id.json')))
    model, device = load_model_and_parallel(model_path, gpu_ids, model_type, strict, bert_dir, label_type_num)
    test_dataset = raw_data_to_dataset(data_dir, bert_dir, 'test.json', 'test', ent2id)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
    if mode == 'evaluate':
        evaluate(model, device, test_loader, ent2id)
    elif mode == 'predict':
        predict(model, device, test_loader, ent2id)
    else:
        logger.error('{} is unknown mode, only support evaluate and predict for now.'.format(mode))
