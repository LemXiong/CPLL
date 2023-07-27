import os
import torch.nn as nn
from transformers import BertModel
from utils.LWLoss import LWLoss_with_cross
import torch


class BaseModel(nn.Module):
    def __init__(self,
                 bert_dir,
                 dropout_prob):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')

        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            f'given path of pretrained bert does not exist, please check out bert_dir:{bert_dir} or config_path:{config_path} '

        self.bert_module = BertModel.from_pretrained(bert_dir,
                                                     output_hidden_states=True,
                                                     hidden_dropout_prob=dropout_prob)

        self.bert_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


class BERTClassifier(BaseModel):
    """
    BERT + Classifier for voted data
    """
    def __init__(self,
                 bert_dir,
                 num_tags,
                 dropout_prob=0.1,
                 **kwargs):
        super(BERTClassifier, self).__init__(bert_dir=bert_dir, dropout_prob=dropout_prob)

        out_dims = self.bert_config.hidden_size

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        self.classifier = nn.Linear(mid_linear_dims, num_tags)

        init_blocks = [self.mid_linear, self.classifier]

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                labels=None,
                mode='train'):

        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        mid_linear_output = self.mid_linear(bert_outputs[0])

        emissions = self.classifier(mid_linear_output)

        if mode == 'train':
            criteria = nn.CrossEntropyLoss()
            tokens_loss = criteria(emissions[attention_masks > 0], labels[attention_masks > 0])
            out = (tokens_loss,)

        else:
            lengths = attention_masks.sum(1).tolist()
            tokens_out = torch.max(emissions, -1).indices.tolist()
            for length, index in zip(lengths, range(len(tokens_out))):
                tokens_out[index] = tokens_out[index][0: int(length)]
            out = (tokens_out, emissions)

        return out


class LW_Model(BaseModel):
    """
    model for crow-annotated data
    """
    def __init__(self,
                 bert_dir,
                 num_tags,
                 dropout_prob=0.1,
                 **kwargs):
        super(LW_Model, self).__init__(bert_dir=bert_dir, dropout_prob=dropout_prob)

        out_dims = self.bert_config.hidden_size

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        self.classifier = nn.Linear(mid_linear_dims, num_tags)

        init_blocks = [self.mid_linear, self.classifier]

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                p_weight=None,
                n_weight=None,
                confidence=None,
                index=None,
                labels=None,
                alpha=0.5,
                mode='cal_loss'):

        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        mid_linear_output = self.mid_linear(bert_outputs[0])

        emissions = self.classifier(mid_linear_output)

        if mode == 'cal_loss':
            tokens_loss = LWLoss_with_cross(emissions, labels, confidence, index, p_weight, n_weight, alpha=alpha)
            out = (tokens_loss,)
        elif mode == 'cal_emissions':
            out = (emissions,)
        else:
            lengths = attention_masks.sum(1).tolist()
            tokens_out = torch.max(emissions, -1).indices.tolist()
            for length, location in zip(lengths, range(len(tokens_out))):
                tokens_out[location] = tokens_out[location][0: int(length)]
            out = (tokens_out, emissions)

        return out
