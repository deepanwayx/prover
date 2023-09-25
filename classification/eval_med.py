import json
import time
import random
import pickle
import glob
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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
class MEDDataset(Dataset):
    def __init__(self, filename):        
        df = pd.read_csv(filename)
        content, labels = list(df["text"]), list(df["labels"])

        self.content, self.labels = content, labels
        
    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        s1, s2 = self.content[index], self.labels[index]
        return s1, s2
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


def configure_dataloaders(batch_size=16):
    "Prepare dataloaders"
    eval_dataset = MEDDataset("data/med.csv")
    eval_loader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, collate_fn=eval_dataset.collate_fn)
    return eval_loader


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
    
    p_score, r_score, _, _ = precision_recall_fscore_support(all_labels_cls, all_preds_cls, labels=[0,1,2], average="weighted", zero_division=0)
    p_score, r_score = round(p_score, 4), round(r_score, 4)
    
    return avg_loss, acc, wf1, mf1, result, p_score, r_score, len(all_labels_cls)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--eval-bs", type=int, default=16, help="Batch size.")
    parser.add_argument("--name", default="microsoft/deberta-v3-large", help="Which model.")
    parser.add_argument("--model-path", default="", help="Trained model path.")
    
    global args
    args = parser.parse_args()
    print(args)
    
    batch_size = args.eval_bs
    model_path = args.model_path
    name = args.name
    
    model = Model(name=name).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    sep_token = model.tokenizer.sep_token
    
    summary = "/".join(model_path.split("/")[:-1]) + "/summary.txt"
    train_config = open(summary).readlines()[0]
    train_exp_id = model_path.split("/")[-2]
    
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/med/" + train_exp_id + "/").mkdir(parents=True, exist_ok=True)
    
    if "/" in name:
        sp = name[name.index("/")+1:]
    else:
        sp = name
    
    rf = "results/med/" + train_exp_id + "/" + sp + ".tsv"
    rf_detailed = "results/med/" + train_exp_id + "/" + "detailed-" + sp + ".tsv"
    
    val_loader = configure_dataloaders(batch_size)

    loss, acc, f1, mf1, cls_results, precision, recall, num_samples = train_or_eval_model(model, val_loader, split="Val")

    with open(rf, "a") as rfx:
        rfx.write("{}\t{}\t{}\t{}\n\n".format(f1, precision, recall, num_samples))

    with open(rf_detailed, "a") as rfdx:
        rfdx.write("F1: {}\tPrecision: {}\tRecall: {}\tSamples: {}\n\n".format(f1, precision, recall, num_samples))
        rfdx.write(cls_results + "\n\n")
