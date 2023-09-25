import json
import time
import random
import pickle
import gc, os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from models import Model
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import Adafactor, get_scheduler

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

class ClassificationDataset(Dataset):
    def __init__(self, filename1, filename2=None):
        df = pd.read_csv(filename1)
        content, labels = list(df["text"]), list(df["labels"])
        if filename2:
            df2 = pd.read_csv(filename2)
            content += list(df2["text"])
            labels += list(df2["labels"])
            
        self.content, self.labels = content, labels
        print (filename1, filename2, dict(pd.Series(labels).value_counts()))
        
    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        s1, s2 = self.content[index], self.labels[index]
        return s1, s2
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


def configure_dataloaders(train_file1, train_file2, train_batch_size=16, eval_batch_size=16):
    "Prepare dataloaders"
    
    train_dataset = ClassificationDataset(train_file1, train_file2)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, collate_fn=train_dataset.collate_fn)

    val_dataset = ClassificationDataset("data/nli/mnli_valid.csv")
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)
    
    test_dataset = ClassificationDataset("data/med.csv")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)

    return train_loader, val_loader, test_loader


def configure_optimizer(model, args):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    return optimizer

def configure_scheduler(optimizer, num_training_steps, args):
    "Prepare scheduler"
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )    
    return lr_scheduler


def train_or_eval_model(model, dataloader, optimizer=None, split="Train"):
    losses, preds_cls, labels_cls,  = [], [], []
    if split=="Train":
        model.train()
    else:
        model.eval()
    
    for batch in tqdm(dataloader, leave=False):
        if split=="Train":
            optimizer.zero_grad()
            
        content, l_cls = batch
        if split=="Train":
            loss, p_cls = model(batch)
        else:
            with torch.no_grad():
                loss, p_cls = model(batch)
        
        preds_cls.append(p_cls)
        labels_cls.append(l_cls)
        
        if split=="Train":
            loss.backward()
            optimizer.step()
            
        losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)
    
    all_preds_cls = [item for sublist in preds_cls for item in sublist]
    all_labels_cls = [item for sublist in labels_cls for item in sublist]
    acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
    wf1 = round(f1_score(all_labels_cls, all_preds_cls, average="weighted"), 4)
    mf1 = round(f1_score(all_labels_cls, all_preds_cls, average="macro"), 4)
    result = str(classification_report(all_labels_cls, all_preds_cls, digits=4))
    
    return avg_loss, acc, wf1, mf1, result

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for transformers.")
    parser.add_argument("--wd", default=0.0, type=float, help="Weight decay for transformers.")
    parser.add_argument("--warm-up-steps", type=int, default=0, help="Warm up steps.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--bs", type=int, default=8, help="Batch size.")
    parser.add_argument("--eval-bs", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs.")
    parser.add_argument("--name", default="microsoft/deberta-v3-large", help="Which model.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--mnli-percent", type=int, default=1, help="Percentage of mnli data")
    parser.add_argument("--aug-percent", type=int, default=0, help="Percentage of augmented data")
    parser.add_argument("--aug", default="None", help="Which augmneted data.")
    
    global args
    args = parser.parse_args()
    print(args)
    
    train_batch_size = args.bs
    eval_batch_size = args.eval_bs
    epochs = args.epochs
    name = args.name
    seed = args.seed
    train_file1 = "data/nli/mnli_{}.csv".format(args.mnli_percent)
    
    if args.aug in ["entailer", "prover"]:
        train_file2 = "data/nli/{}_{}.csv".format(args.aug, args.aug_percent)
    else:
        train_file2 = None
    
    np.random.seed(seed); random.seed(seed)
    
    model = Model(name=name).cuda()
    
    sep_token = model.tokenizer.sep_token
    
    train_loader, val_loader, test_loader = configure_dataloaders(
        train_file1, train_file2, train_batch_size, eval_batch_size
    )
    
    optimizer = configure_optimizer(model, args)
    
    if "/" in name:
        sp = name[name.index("/")+1:]
    else:
        sp = name
    
    exp_id = str(int(time.time()))
    vars(args)["exp_id"] = exp_id
    
    path = "saved/" + exp_id + "/" + name.replace("/", "-")
    Path("saved/" + exp_id + "/").mkdir(parents=True, exist_ok=True)
    Path("results/").mkdir(parents=True, exist_ok=True)
    
    fname = "saved/" + exp_id + "/" + "summary.txt"
    
    f = open(fname, "a")
    f.write(str(args) + "\n\n")
    f.close()
        
    lf_name = "results/" + name.replace("/", "-") + ".txt"
    lf_all = str(args) + "\n\n"
    
    best_val_loss, best_val_f1 = np.inf, 0
    for e in range(epochs):        
        train_loss, train_acc, train_f1, train_mf1, _ = train_or_eval_model(model, train_loader, optimizer, "Train")
        val_loss, val_acc, val_f1, val_mf1, val_results = train_or_eval_model(model, val_loader, split="Val")
        test_loss, test_acc, test_f1, test_mf1, test_results = train_or_eval_model(model, test_loader, split="Test")
        
        x = "Epoch {}: Loss: Train {}; Val {}".format(e+1, train_loss, val_loss)
        y = "MNLI Results: \n{}".format(val_results)
        z = "MED Results: \n{}".format(test_results)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "saved/" + exp_id + "/best_loss.pt")
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "saved/" + exp_id + "/best_f1.pt")
            
        print (x)
        print (y)
        print (z)
        
        lf_all += x + "\n" + y + "\n" + z + "\n\n"

        f = open(fname, "a")
        f.write(x + "\n" + y + "\n" + z + "\n\n")
        f.close()
        
    lf = open(lf_name, "a")
    lf.write(lf_all + "-"*100 + "\n")
    lf.close()
