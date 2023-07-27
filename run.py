import json
import logging
import os
import sys
import time

import torch
from datetime import timedelta
from utils.test import test
from utils.processor import raw_data_to_dataset
from utils.args import get_args, print_args
from utils.trainer import Trainer
from utils.set_seed import set_seed
from model.model import *


def run():
    torch.set_printoptions(profile='full', precision=8, sci_mode=False, linewidth=200)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    start_time = time.time()
    logger.info('----------------timing----------------')

    args = get_args()
    args.output_dir = os.path.join(args.output_dir, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    print_args(args)

    set_seed(args.seed)

    ent2id = json.load(open(os.path.join(args.ent2id_dir, args.ent2id_way + '_ent2id.json')))
    model = eval(args.model + '(args.bert_dir, len(ent2id))')

    train_dataset = raw_data_to_dataset(args.data_dir, args.bert_dir, 'train.json', 'train', ent2id)
    dev_dataset = None
    if not args.no_eval:
        dev_dataset = raw_data_to_dataset(args.data_dir, args.bert_dir, 'dev.json', 'dev', ent2id)
    trainer = Trainer(model)
    best_step = trainer.train(args, train_dataset, dev_dataset=dev_dataset, ent2id=ent2id)

    if args.do_test:
        logger.info('----------------start testing----------------')
        model_path = os.path.join(args.output_dir, 'step-{}.pt'.format(best_step))
        test(model_path, args.model, args.gpu_ids, args.data_dir, ent2id, args.bert_dir, len(ent2id),
             batch_size=args.test_batch_size)
        logger.info('----------------test done----------------')

    end_time = time.time()
    logger.info("------------ time cost: {}-----------".format(timedelta(seconds=int(round(end_time-start_time)))))


if __name__ == '__main__':
    run()
    