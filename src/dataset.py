import config
from torch.utils.data import DataLoader
import pandas as pd
import torch

class BERTDataset:
    """
    Handle both Train and Test data.
    """
    def __init__(self, q1, ans, label = None):
        self.q1 = q1
        self.ans = ans
        self.label = label
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.ans)

    def __getitem__(self, idx):
        qn1 = str(self.q1[idx])
        qn1 = " ".join(qn1.split())
        ans = str(self.ans[idx])
        ans = " ".join(ans.split())

        inputs = self.tokenizer(
                qn1,
                ans, 
                return_tensors="pt",
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

        if self.label is not None:
            label = self.label[idx]
            res["target"] = torch.tensor(label, dtype = torch.long)

        return res

if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    dataset = BERTDataset(
        q1=df.Question.values, ans=df.Answer.values, label = df.Label.values
    )
    data_loader = DataLoader(
        dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    for d in data_loader:
        print(d)
