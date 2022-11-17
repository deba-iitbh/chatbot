import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertForNextSentencePrediction.from_pretrained(config.BERT_PATH)

    def forward(self, ids, mask, token_type_ids, labels):
        out = self.bert(input_ids = ids, attention_mask=mask, token_type_ids=token_type_ids, labels = labels)
        return out.loss, out.logits[1]
