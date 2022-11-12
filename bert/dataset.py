import config
import torch

class BERTDataset:
    def __init__(self, qn, ans):
        self.q = qn
        self.ans = ans
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        qn = str(self.q[idx])
        qn = " ".join(qn.split())

        ans = str(self.ans[idx])
        ans = " ".join(ans.split())

        inputs = self.tokenizer.encode_plus(
            qn,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )

        targets = self.tokenizer.encode_plus(
            ans,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )

        q_ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        t_ids = targets["input_ids"]

        return {
            "ids": torch.tensor(q_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(t_ids, dtype=torch.long),
        }
