import json

import torch
import numpy as np
import logging
from utils.decode import decode


logger = logging.getLogger(__name__)


def predict(model, device, data_loader, ent2id, save_result=True, result_path='./predict.json'):
    model.eval()

    id2ent = {}
    for ent in ent2id:
        id2ent.update({
            ent2id[ent]: ent
        })

    with torch.no_grad():
        results = []
        for index, batch_data in enumerate(data_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            batch_preds = model(mode='evaluate', **batch_data)[0]
            for token_id, pred in zip(batch_data['token_ids'], batch_preds):
                tokens = []
                labels = []
                for location in range(0, len(pred)):
                    tokens.append(int(token_id[location]))
                    labels.append(id2ent[pred[location]])
                    logger.info('{} {}'.format(token_id[location], id2ent[pred[location]]))
                assert token_id[len(pred)-1] == 102, 'Seem likes there are abnormal [SEP].'
                result = {'tokens': tokens,
                          'labels': labels}
                results.append(result)
        if save_result:
            with open(result_path, 'w', encoding='utf-8') as f_out:
                json.dump(results, f_out, ensure_ascii=False)
