import config
import dataset
import engine
import torch
import pandas as pd
import logging
logging.disable(logging.INFO)

from model import BERTBaseUncased
from torch.utils.data import DataLoader
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_constant_schedule_with_warmup
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
        qn=df_train.Question.values, ans=df_train.Answer.values
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        qn=df_valid.Question.values, ans=df_valid.Answer.values
    )

    valid_data_loader = DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    print(f"Len of Train Dataset: {len(train_dataset)}")
    print(f"Len of Valid Dataset: {len(valid_dataset)}")

    device = torch.device(config.DEVICE)
    model = BERTBaseUncased()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=5, last_epoch=-1
    )

    best_eval_loss = float('inf')
    for _ in trange(config.EPOCHS, unit="epoch"):
        total_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        print(total_loss)
        eval_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Eval Loss = {eval_loss}")
        if eval_loss < best_eval_loss:
            # torch.save(model.state_dict(), config.MODEL_PATH)
            best_eval_loss = eval_loss

def run_inference(q):
    inf_dataset = dataset.BERTDataset(
        qn=pd.Series(q)
    )
    inf_data_loader = DataLoader(
        inf_dataset, batch_size=1
    )

    device = torch.device(config.DEVICE)
    model = BERTBaseUncased()
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    res = engine.inference_fn(inf_data_loader, model, device)
    return res

if __name__ == "__main__":
    run()

    # q = "Location of IIT Bhilai?"
    # print(run_inference(q))
