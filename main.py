
from dataset import RecSysDataset
from train import Trainer
from models import VBPR
import torch

if __name__ == '__main__':
    k=10
    k2=20
    batch_size=128
    n_epochs=20
    dataset = RecSysDataset()
    vbpr = VBPR(
        dataset.n_users, dataset.n_items, 
        dataset.corpus.image_features, k, k2)
    tr = Trainer(vbpr, dataset)
    tr.train(n_epochs, batch_size)

    torch.save(vbpr, 'vbpr_resnet50_v1.pth')