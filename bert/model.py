import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, config.MAX_LEN)

    def forward(self, ids, mask, token_type_ids):
        o = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        out = o["pooler_output"]
        bo = self.bert_drop(out)
        output = self.out(bo)
        return output
