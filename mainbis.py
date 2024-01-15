from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
import argparse
from pathlib import Path

# custom librairies
from libraries.loss import ConstrastiveLoss
from libraries.dataloader import GraphTextDataset, GraphDataset, TextDataset
from libraries.model import Model
from libraries.check_submission import compute_test_mrr
from libraries.train import train, create_submission

def launch_experiment(args):
    
    # check if fresh experiment
    if args.ckpt == None:
        resume = False
    else:
        resume = True
    experiment_path = Path(Path.cwd(), args.exp_name)
    if not experiment_path.is_dir():
        os.mkdir(experiment_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    if not resume:
        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        
    else:
        checkpoint = torch.load(args.ckpt)
        model = Model(model_name=args.model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model.to(device)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    loss_fn = ConstrastiveLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.999),
                            weight_decay=0.01)
    
    best_model_path = train(model, optimizer, loss_fn, train_loader, val_loader, device, experiment_path, args)
    
    create_submission(best_model_path, gt, tokenizer, device, experiment_path, args)
    
    print(compute_test_mrr(Path(experiment_path, 'submission.csv')))
    

if __name__ == '__main__':
    # parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", type=int, help="number of epochs")
    parser.add_argument("--batch_size", type=int, help="batch size for experiments")
    parser.add_argument("--lr", type=float, help="learning_rate")
    parser.add_argument("--exp_name", type=str, help="name of the experiment")
    parser.add_argument("--ckpt", type=str, help="checkpoint of the experiment")

    args = parser.parse_args()
            
    launch_experiment(args)
    