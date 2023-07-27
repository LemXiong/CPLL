import sys

from transformers import BertTokenizer
import json
import os
import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self,
                 set_type,
                 tokens,
                 labels=None,
                 max_tokens_length=510):
        self.set_type = set_type
        self.tokens = tokens
        self.labels = labels
        if len(tokens) > max_tokens_length:
            self.tokens = self.tokens[0:max_tokens_length]
            if type(labels[0]) == list:
                for index, label in enumerate(labels):
                    self.labels[index] = label[0:max_tokens_length]
            elif type(labels[0]) == str:
                self.labels = self.labels[0:max_tokens_length]
        
        
class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.labels = labels


class NERDataset(Dataset):
    def __init__(self, features):

        self.nums = len(features)

        self.token_ids = [torch.tensor(feature.token_ids).long() for feature in features]
        self.attention_masks = [torch.tensor(feature.attention_masks).float() for feature in features]
        self.token_type_ids = [torch.tensor(feature.token_type_ids).long() for feature in features]
        self.labels = [torch.tensor(feature.labels) for feature in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}

        if self.labels is not None:
            data['labels'] = self.labels[index]

        return data


def process_partial_label(partial_label, ent2id, max_seq_len):
    res = [[0.] * len(ent2id) for i in range(len(partial_label[0]))]
    for label in partial_label:
        for index, token in enumerate(label):
            res[index][ent2id[token]] += 1.
    res.insert(0, [float(len(partial_label))] + [0.] * (len(ent2id)-1))     # CLS
    while len(res) < max_seq_len:
        res.append([float(len(partial_label))] + [0.] * (len(ent2id)-1))    # SEP and PAD
    return res


def read_json(file_path):
    with open(file_path, encoding='utf-8') as f:
        raw_examples = json.load(f)
    return raw_examples


def get_examples(raw_examples, set_type):
    examples = []

    for i, item in enumerate(raw_examples):
        examples.append(InputExample(set_type=set_type,
                                     tokens=item['tokens'],
                                     labels=(item['labels'] if 'labels' in item else None)))
    return examples


def char_level_tokenize(raw_tokens, tokenizer):
    tokens = []

    for raw_token in raw_tokens:
        if raw_token in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(raw_token)):
                tokens.append('[INV]')
            else:
                tokens.append(raw_token)

    return tokens
    
    
def convert_examples_to_features(bert_dir, examples, ent2id, max_seq_len=512):
    features = []

    logger.info(f'Convert {len(examples)} {examples[0].set_type} examples to features')
    for example in examples:
        raw_tokens = example.tokens
        labels = example.labels
        tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))

        tokens = char_level_tokenize(raw_tokens, tokenizer)
        assert len(tokens) == len(raw_tokens)

        label_ids = None
        if labels is not None:
            if type(labels[0]) == list:
                label_ids = process_partial_label(labels, ent2id, max_seq_len)
            else:
                label_ids = [0]
                for label in labels:
                    label_ids.append(ent2id[label])
                label_ids.append(0)

                # pad
                if len(label_ids) < max_seq_len:
                    pad_length = max_seq_len - len(label_ids)
                    label_ids = label_ids + [0] * pad_length

            assert len(label_ids) == max_seq_len, 'label padding go wrong'

        encode_dict = tokenizer.encode_plus(text=tokens,
                                            max_length=max_seq_len,
                                            truncation=True,
                                            padding='max_length',
                                            is_pretokenized=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        token_ids = encode_dict['input_ids']
        attention_masks = encode_dict['attention_mask']
        token_type_ids = encode_dict['token_type_ids']

        features.append(BaseFeature(
            # bert inputs
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids,
            labels=label_ids,
        ))

    logger.info(f'Build {len(features)} features')
    return features
    
    
def raw_data_to_dataset(data_dir, bert_dir, filename, set_type, ent2id):
    examples = read_json(os.path.join(data_dir, filename))
    examples = get_examples(examples, set_type)
    features = convert_examples_to_features(bert_dir, examples, ent2id)
    dataset = NERDataset(features)
    return dataset

