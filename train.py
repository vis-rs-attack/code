import numpy as np
import torch.nn as nn
import torch
import time

from argparse import ArgumentParser

from dataset import RecSysDataset
from models import VBPR, VBPRC, DeepStyle, BPR

from torch.utils.data import DataLoader

import utils

tqdm = utils.import_tqdm()

class Trainer:
  def __init__(self, net, dataset):
    self.net = net
    self.dataset = dataset
    self.n_users = dataset.n_users
    self.n_items = dataset.n_items

  def evaluate(self, batch_size=2000, sample_size=-1, cold_start=False):
    self.net.eval()
    sum_loss = 0.
    sum_auc = 0.
    count = 0
    validset = self.dataset.get_validation(sample_size)
    loader = DataLoader(validset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        loss, auc = self.net(*batch)
        sum_auc += auc
        sum_loss += loss.data.item()
        count += len(batch[0])
    return sum_loss/count, sum_auc/count
  
  def train(self, n_epochs=50, batch_size=2000, patience=2, save_best=True, learning_rate=0.01):
    if save_best:
      timestr = time.strftime("%y%m%d-%H%M")
      best_model_filename = f'best_{timestr}.pth'
    loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    best_val_auc = 0
    no_change = 0
    for epoch in range(1, n_epochs+1):
        self.net.train()
        start = time.time()
        sum_loss = 0.
        sum_auc = 0.
        count = 0
        for batch in tqdm(loader):

            loss, auc = self.net(*batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_auc += auc
            sum_loss += loss.data.item()
            count += len(batch[0])

        elapsed = time.time() - start

        train_loss, train_auc = sum_loss/count, sum_auc/count
        val_loss, val_auc = self.evaluate()

        print(f'{epoch}\t{elapsed:.2f}\t{train_loss:.4f}\t{train_auc:.4f}\t{val_loss:.4f}\t{val_auc:.4f}')

        if val_auc > best_val_auc:
          if save_best:
            torch.save(self.net.state_dict(), best_model_filename)
          best_val_auc = val_auc
          no_change = 0
        else:
          no_change += 1
          if no_change > patience:
            print(f"early stopping.")
            break

if __name__ == '__main__':
  parser = ArgumentParser(description="Trainging")

  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--k', type=int, default=10)
  parser.add_argument('--k2', type=int, default=10)
  parser.add_argument('--batch_size', type=int, default=2000)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--learning_rate', type=float, default=0.01)
  parser.add_argument('--patience', type=int, default=2)
  parser.add_argument('--lambda_w', type=float, default=0.01)
  parser.add_argument('--lambda_b', type=float, default=0.01)
  parser.add_argument('--lambda_e', type=float, default=0.0001)
  parser.add_argument('--algorithm', type=str, default='deepstyle') # vbpr, vbprc, deepstyle
  parser.add_argument('--dataset', type=str, default='Electronics')

  args = parser.parse_args()

  print(args)

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  dataset = RecSysDataset(args.dataset)
  if args.algorithm == "vbpr":
    model = VBPR(
        dataset.n_users, dataset.n_items, dataset.corpus.image_features, 
        args.k, args.k2, args.lambda_w, args.lambda_b, args.lambda_e)
  elif args.algorithm == "vbprc":
    model = VBPRC(
        dataset.n_users, dataset.n_items, dataset.n_categories, 
        dataset.corpus.image_features, dataset.corpus.item_category,
        args.k, args.k2, args.lambda_w, args.lambda_b, args.lambda_e)
  elif args.algorithm == "deepstyle":
    model = DeepStyle(
        dataset.n_users, dataset.n_items, dataset.n_categories, 
        dataset.corpus.image_features, dataset.corpus.item_category,
        args.k, args.lambda_w, args.lambda_e)
  elif args.algorithm == "bpr":
    model = BPR(dataset.n_users, dataset.n_items, args.k, args.lambda_w, args.lambda_b)

  if torch.cuda.is_available():
      model = model.cuda()

  tr = Trainer(model, dataset)
  tr.train(args.epochs, args.batch_size, args.patience, args.learning_rate)

   
