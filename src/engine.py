import torch
from config import TOKENIZER
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    pgbar = tqdm(data_loader, total=len(data_loader), unit="batch")
    avg_loss = 0
    for d in pgbar:
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["target"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        loss, score = model(ids=ids, mask=mask, token_type_ids=token_type_ids, labels = targets)
        pgbar.set_postfix(loss = loss.item())
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return avg_loss/len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    pgbar = tqdm(data_loader, total=len(data_loader), unit="batch")
    avg_loss = 0
    with torch.no_grad():
        for d in pgbar:
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["target"]
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)
            loss, score = model(ids=ids, mask=mask, token_type_ids=token_type_ids, labels = targets)
            print(score)
            pgbar.set_postfix(loss = loss.item())
            avg_loss += loss.item()
    return avg_loss/len(data_loader)

def inference_fn(data_loader, model, device):
    model.eval()
    fin_outputs = []
    pgbar = tqdm(data_loader, total=len(data_loader), unit="batch")
    with torch.no_grad():
        for d in pgbar:
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            fin_outputs.extend(outputs)
    return fin_outputs
