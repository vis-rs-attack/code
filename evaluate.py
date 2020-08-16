from argparse import ArgumentParser

from train import Trainer
from dataset import RecSysDataset
from models import VBPR, VBPRC, DeepStyle, BPR

if __name__ == '__main__':
    parser = ArgumentParser(description="Experiments")

    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--k2', type=int, default=10)
    parser.add_argument('--algorithm', type=str, default='deepstyle') # bpr, vbpr, vbprc, deepstyle
    parser.add_argument('--dataset', type=str, default='Electronics')

    args = parser.parse_args()
    
    print(args)

    dataset = RecSysDataset(args.dataset)
    if args.algorithm == "vbpr":
        model = VBPR(
            dataset.n_users, dataset.n_items, dataset.corpus.image_features, 
            args.k, args.k2)

    elif args.algorithm == "vbprc":
        model = VBPRC(
            dataset.n_users, dataset.n_items, dataset.n_categories, 
            dataset.corpus.image_features, dataset.corpus.item_category,
            args.k, args.k2)

    elif args.algorithm == "deepstyle":
        model = DeepStyle(
            dataset.n_users, dataset.n_items, dataset.n_categories, 
            dataset.corpus.image_features, dataset.corpus.item_category, args.k)

    elif args.algorithm == "bpr":
        model = BPR(dataset.n_users, dataset.n_items, args.k)

    model.load(f'../data/dataset/{args.dataset}/models/{args.algorithm}_resnet50.pth')

    tr = Trainer(model, dataset)

    print(tr.evaluate())
