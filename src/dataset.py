import config
import torch

class BERTDataset:
    """
    Handle both Train and Test data.
    """
    def __init__(self, qn, ans=None):
        self.q = qn
        self.ans = ans
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        qn = str(self.q[idx])
        qn = " ".join(qn.split())

        inputs = self.tokenizer.encode_plus(
            qn,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )

        q_ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        res = {
            "ids": torch.tensor(q_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

        if self.ans is not None:
            ans = str(self.ans[idx])
            ans = " ".join(ans.split())
            targets = self.tokenizer.encode_plus(
                ans,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
            )
            t_ids = targets["input_ids"]
            res["targets"] = torch.tensor(t_ids, dtype=torch.long)

        return res
