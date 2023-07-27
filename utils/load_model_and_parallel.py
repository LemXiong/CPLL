import torch
import logging
from model.model import *

logger = logging.getLogger(__name__)


def load_model_and_parallel(model, gpu_ids, model_type=None, strict=True, bert_dir=None, label_type_num=None):
    gpu_ids = gpu_ids.split(',')

    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    if type(model) == str:
        assert bert_dir is not None, 'locally load model need to assign bert dir'
        assert label_type_num is not None, 'locally load model need to assign label_type_num file'
        assert model_type is not None, 'locally load model must assign model type'
        logger.info(f'Load ckpt from {model}')
        ckpt_path = model
        if type(label_type_num) == str:
            label_type_num = int(label_type_num)
        model = eval(model_type + '(bert_dir, label_type_num)')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

    model.to(device)

    if len(gpu_ids) > 1:
        logger.info(f'Use multi gpus in: {gpu_ids}')
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        logger.info(f'Use single gpu in: {gpu_ids}')

    return model, device
