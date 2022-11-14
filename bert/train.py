import config
import dataset
import engine
import torch
import pandas as pd

from model import BERTBaseUncased
from torch.utils.data import DataLoader
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import trange

import warnings
warnings.filterwarnings("ignore")


def run():
    df = pd.read_csv(config.TRAINING_FILE)
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.1, random_state=42,
    )

    df_train:pd.DataFrame = df_train.reset_index(drop=True)
    df_valid:pd.DataFrame = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        qn=df_train.q.values, ans=df_train.a.values
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        qn=df_valid.q.values, ans=df_valid.a.values
    )

    valid_data_loader = DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    print(f"Len of Train Dataset: {len(train_dataset)}")
    print(f"Len of Valid Dataset: {len(valid_dataset)}")

    device = torch.device(config.DEVICE)
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # best_accuracy = 0
    for _ in trange(config.EPOCHS, unit="epoch"):
        total_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        print(total_loss)
        # outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        # outputs = np.array(outputs) >= 0.5
        # accuracy = metrics.accuracy_score(targets, outputs)
        # print(f"Accuracy Score = {accuracy}")
        # if accuracy > best_accuracy:
        #     torch.save(model.state_dict(), config.MODEL_PATH)
        #     best_accuracy = accuracy


if __name__ == "__main__":
    run()
