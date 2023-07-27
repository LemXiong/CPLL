import argparse
import logging

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # path args
    parser.add_argument('--bert_dir', default='pretrain/torch_roberta_wwm', type=str)

    parser.add_argument('--data_dir', default='/workspace/data/resume_partial', type=str)

    parser.add_argument('--ent2id_dir', default='/workspace/data/resume_partial', type=str)

    parser.add_argument('--output_dir', default='out', type=str)
    
    # train args
    parser.add_argument('--warmup_proportion', default=0.1, type=float)

    parser.add_argument('--weight_decay', default=0., type=float)

    parser.add_argument('--train_epochs', default=10, type=int)

    parser.add_argument('--alpha', default=0.5, type=float,
                        help='balance between confidence and label num')

    parser.add_argument('--batch_size', default=8, type=int)

    parser.add_argument('--bert_lr', default=2e-5, type=float,
                        help='learning rate for bert module')
    
    parser.add_argument('--other_lr', default=2e-3, type=float,
                        help='learning rate for modules except bert')
    
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='be used to clip grad')

    parser.add_argument('--p_weight', default=1.0, type=float,
                        help='weight in LWLoss for positive sample')

    parser.add_argument('--n_weight', default=2.0, type=float,
                        help='weight in LWLoss for negative sample, Namely beta in the equation')
    
    # dev args
    parser.add_argument('--no_eval', default=False, action='store_true',
                        help='whether to eval the model, the model will only be saved in the last step if it is false')

    parser.add_argument('--eval_step', default=100, type=int,
                        help='model will be evaluated every eval_step')

    parser.add_argument('--eval_criterion', default='f1', choices=['f1', 'precision', 'recall'], type=str,
                        help='best model will be the one have the highest eval_criterion')
    
    # test args
    parser.add_argument('--do_test', default=False, action='store_true')

    parser.add_argument('--test_batch_size', default=64, type=int)
    
    # other args
    parser.add_argument('--model', default='LW_Model', type=str, choices=['LW_Model', 'BERTClassifier'])

    parser.add_argument('--no_save', default=False, action='store_true')

    parser.add_argument('--no_delete_not_optimal_model', default=False, action='store_true',
                        help='keep all the model evaluated')

    parser.add_argument('--seed', default=1248, type=int,
                        help='random seed')

    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='assign which gpu is going to be used, -1 for cpu, "0,1" for multi gpu')

    parser.add_argument('--ent2id_way', default='bio', type=str)

    return parser.parse_args()


def print_args(args):
    arg_dict = vars(args)
    logger.info('----------------start printing args----------------')
    for key in arg_dict:
        logger.info('{}: {}'.format(key, arg_dict[key]))
    logger.info('------------------finish printing------------------')
