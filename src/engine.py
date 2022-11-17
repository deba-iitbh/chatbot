import torch
import numpy as np
from config import TOKENIZER
import torch.nn as nn
from tqdm import tqdm


def loss_fn(outputs, targets):
    mseloss = nn.MSELoss()
    return torch.sqrt(mseloss(outputs.float(), targets.float()))


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    pgbar = tqdm(data_loader, total=len(data_loader), unit="batch")
    avg_loss = 0
    for d in pgbar:
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
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
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(outputs, targets)
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
            final_output = outputs.cpu().detach().numpy()[0]
            final_output = final_output.astype(int) + 1000
            final_output = final_output.tolist()

            # Detokenize
            tokens = TOKENIZER.convert_ids_to_tokens(final_output)
            print(tokens)
            text = ' '.join([x for x in tokens])
            fine_text = text.replace(' ##', '')
            final_output.extend(fine_text)
    return fin_outputs
