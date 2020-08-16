import os
import numpy as np

from dataset import RecSysDataset, read_item_data, read_user_item_data
from models import VBPR, VBPRC, DeepStyle, ImageModel
import utils
import numbers
import random

import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
from PIL import Image

import pickle, json, time

from argparse import ArgumentParser

from numpy.linalg import norm
import scipy.stats

tqdm = utils.import_tqdm()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def cosine(a,b):
    return np.dot(a,b)/norm(b)/norm(a)

def to_img(x):
    x = x.squeeze(0).type(torch.DoubleTensor)
    x = x.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()
    x = np.transpose(x*255, (1, 2, 0))
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

from sklearn import linear_model, svm

def solve(W, y):
    # x, _, _, _ = np.linalg.lstsq(W, y, rcond=None)
    x = linear_model.LinearRegression().fit(W, y).coef_
    return x

def pad_resize(im, desired_size=224):
    old_size = im.size

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_size, desired_size), "white")
    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

    return new_im

class TorchPCA:
    def __init__(self, weight, bias):
        self.W = torch.tensor(weight.T, dtype=torch.float32)
        self.b = torch.tensor(bias, dtype=torch.float32)

    def transform(self, X):
        return torch.mm(X-self.b, self.W)

    @staticmethod
    def get_pca(X, n_components):

        from sklearn.decomposition import PCA

        filename = f"../data/pca_{n_components}.pkl"

        if not os.path.exists(filename):
            pca = PCA(n_components=n_components)
            pca.fit(X)

            W, b = pca.components_, pca.mean_

            with open(filename, "bw") as f:
                pickle.dump((W, b), f) 
        else:
            with open(filename, "br") as f:
                W, b = pickle.load(f)
        
        return TorchPCA(W, b)

class JsonLooger():
    import json
    
    def __init__(self, logfile):
        self.logfile = logfile
        
    def __enter__(self):
        self.fd = open(self.logfile, "w")
        return self
    
    def __exit__(self, type, value, traceback):
        self.fd.close()
        
    def log(self, d):
        json.dump(d, self.fd)
        self.fd.write("\n")

class Experimentation:
    def __init__(self, model, dataset_name):

        dataset_path = f"../data/dataset/{dataset_name}"

        items_data, items_idx = read_item_data(dataset_path)
        user_items, _ = read_user_item_data(dataset_path)

        self.model = model
        self.image_model = ImageModel('resnet50')
        self.user_items = user_items
        self.items_data = items_data
        self.items_idx = items_idx

        self.n_users = model.n_users
        self.n_items = model.n_items

        self.image_folder = dataset_path + "/images"

    # def score_rank(self, rank):
    #     return 1-rank/(self.n_items+1)

    def score_rank(self, rank, rank_distribution="normal"):
        quantile = 1-rank/(self.n_items+1)
        if rank_distribution == "normal":
            return scipy.stats.norm.ppf(quantile)
        return quantile

    def search(self, u, n=10):
        res = self.model.score(u)
        res = res.argsort(dim=0, descending=True).squeeze().numpy()
        return res[:n]

    def load_image(self, i):
        name = self.items_data[i][2]
        filename = os.path.join(self.image_folder, name)
        img = Image.open(filename)
        return pad_resize(img)

    def get_image_features(self, img):
        return self.image_model.get_features(img)/60

    def get_rank(self, u_scores, score):
        return int(np.searchsorted(-u_scores, -score)) + 1

    def run_wb_single_user(self, args, logger):

        u = args.user
        steps = args.steps
        epsilon = int(args.epsilon * 255)

        if hasattr(args, 'item') and args.item is not None:
            t = args.item
            result = self.search(u, -1)
            from_rank = np.where(result==t)[0][0].item() + 1
        else:
            from_rank = args.from_rank
            result = self.search(u, from_rank)
            t = result[from_rank-1].item()

        print(f"Run single user, white-box, for user {u} from rank {from_rank}")

        logger.log({
            'user': u,
            'item': t,
            'rank': from_rank,
        })

        u_scores = self.model.score(u).detach().numpy().squeeze()
        u_scores.sort()
        u_scores = np.flip(u_scores)

        img = self.load_image(t)
        img_M = np.array(img)

        for step in range(steps):
            t_img_tran = self.image_model.transform(img)
            t_img_tran = torch.unsqueeze(t_img_tran, 0)
            t_img_var = nn.Parameter(t_img_tran)
            t_feat = self.get_image_features(t_img_var)

            zero_gradients(t_img_var)
            loss = self.model.pointwise_forward([u], [t], t_feat)
            loss.backward(retain_graph=True)

            grad = t_img_var.grad.data
            grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
            img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_M)

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))
            score = self.model.score_user_item(u, t, feat_t.detach()).item()
            rank = self.get_rank(u_scores, score)

            print(f"Step: {step+1}, rank={rank}")

            logger.log({
                'step': step,
                'rank': rank,
            })

            if rank == 1:
                break
            
        return img

    def run_wb_single_user_o(self, args, logger):

        u = args.user
        steps = args.steps
        epsilon = args.epsilon

        if hasattr(args, 'item') and args.item is not None:
            t = args.item
            result = self.search(u, -1)
            from_rank = np.where(result==t)[0][0].item() + 1
        else:
            from_rank = args.from_rank
            result = self.search(u, from_rank)
            t = result[from_rank-1]

        print(f"Run single user, white-box, for user {u} from rank {from_rank}")

        t_img = self.load_image(t)

        t_img_tran = self.image_model.transform(t_img)
        t_img_tran = torch.unsqueeze(t_img_tran, 0)
        t_img_var = nn.Parameter(t_img_tran)
        t_feat = self.get_image_features(t_img_var)

        u_scores = self.model.score(u).detach().numpy().squeeze()
        u_scores.sort()
        u_scores = np.flip(u_scores)

        for step in range(steps):
            zero_gradients(t_img_var)
            loss = self.model.pointwise_forward([u], [t], t_feat)
            score = -loss.item()
            rank = self.get_rank(u_scores, score)
            print(f"Step: {step+1}, Rank: {rank}")

            logger.log({
                'step': step+1,
                'rank': rank,
            })

            if rank == 1:
                break

            loss.backward(retain_graph=True)
            grad = torch.sign(t_img_var.grad.data)
            adversarial = t_img_var.data - epsilon * grad
            t_img_var.data = adversarial
            t_feat = self.get_image_features(t_img_var)

        return Image.fromarray(to_img(adversarial))

    def bb_attack(
            self, u, t, img, img_M, ori_img_M, n_examples, n_features, u_scores, 
            do_pca, pca_fc, by_rank, rank_distribution, epsilon, eps, gamma, attack="ifgsm"
        ):
        img_t = self.image_model.transform(img)
        img_t = img_t.unsqueeze(0)
        var_t = nn.Parameter(img_t)
        feat_t = self.get_image_features(var_t)

        y_t = self.model.score_user_item(u, t, feat_t.detach()).item()

        rank = self.get_rank(u_scores, y_t)

        if do_pca:
            pca_t = pca_fc.transform(feat_t)

        if by_rank:
            s_t = self.score_rank(rank, rank_distribution)

        W = np.zeros((n_examples, n_features), dtype=np.float32)
        y = np.zeros(n_examples, dtype=np.float32)

        for i in tqdm(range(n_examples)):
            d = np.random.choice(range(-gamma, gamma+1), size=img_M.shape)
            img_i = np.clip(img_M.astype(np.int) + d, 0, 255).astype(np.uint8)
            img_i = Image.fromarray(img_i)

            img_i = self.image_model.transform(img_i)
            feat_i = self.get_image_features(img_i.unsqueeze(0))
            if do_pca:
                pca_i = pca_fc.transform(feat_i)

            y_i = self.model.score_user_item(u, t, feat_i).item()

            if by_rank:
                s_i = self.score_rank(self.get_rank(u_scores, y_i))

            diff = pca_i-pca_t if do_pca else feat_i-feat_t

            W[i] = diff.detach().numpy()
            y[i] = s_t-s_i if by_rank else y_t-y_i

        x = solve(W, y)
        
        if do_pca:
            pca_t.backward(torch.tensor(x).view(1, n_features), retain_graph=True)
        else:
            feat_t.backward(torch.tensor(x).view(1, n_features), retain_graph=True)

        grad = var_t.grad

        # fgsm, ifgsm, pgd, first
        if attack == "first":
            grad = grad.squeeze(0).numpy().transpose((1, 2, 0))
            grad /= norm(grad)
            grad = np.clip(grad*255, -epsilon, epsilon)
            img_M = np.clip(img_M.astype(np.int) - grad, 0, 255).astype(np.uint8)
        elif attack == "ifgsm":
            grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
            img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)
        elif attack == "pgd":
            grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
            img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255)
            eta = np.clip(img_M - ori_img_M, -eps, eps)
            img_M = np.clip(ori_img_M+eta, 0, 255).astype(np.uint8)

        return img_M

    def build_eq_attach(self, u, t, img_M, feat_t, rank, u_scores, n_examples, n_features, gamma):
        W = np.zeros((n_examples, n_features), dtype=np.float32)
        y = np.zeros(n_examples, dtype=np.float32)

        s_t = self.score_rank(rank)

        for i in tqdm(range(n_examples)):
            d = np.random.choice(range(-gamma, gamma+1), size=img_M.shape)
            img_i = np.clip(img_M.astype(np.int) + d, 0, 255).astype(np.uint8)
            img_i = Image.fromarray(img_i)

            img_i = self.image_model.transform(img_i)
            feat_i = self.get_image_features(img_i.unsqueeze(0))

            y_i = self.model.score_user_item(u, t, feat_i).item()

            s_i = self.score_rank(self.get_rank(u_scores, y_i))

            diff = feat_i-feat_t

            W[i] = diff.detach().numpy()
            y[i] = s_t-s_i

        return W, y

    def bb_attack_limitted_returned_items(
            self, u, u_i, u_tag, t, img, n, n_examples, n_features, epsilon, gamma
        ):

        max_items = len(self.items_data)//100
        
        img_M = np.array(img)
        var_t = self.image_model.transform(img).unsqueeze(0)
        feat_t = self.get_image_features(var_t)

        def get_u_scores(u):
            u_scores = self.model.score(u).detach().numpy().squeeze()
            u_scores.sort()
            return np.flip(u_scores)

        u_scores = get_u_scores(u)
        u_i_scores = get_u_scores(u_i)
        u_tag_scores = get_u_scores(u_tag)

        y_ut = self.model.score_user_item(u, t, feat_t).item()
        rank = self.get_rank(u_scores, y_ut)

        # if rank <= 1:
        #     print(f"Rank: {rank}")
        #     return img_M, rank

        def real_grad(u, t):
            var_t = self.image_model.transform(img).unsqueeze(0)
            feat_t = nn.Parameter(self.get_image_features(var_t))

            zero_gradients(feat_t)
            loss = self.model.pointwise_forward([u], [t], feat_t)
            loss.backward(retain_graph=True)

            return feat_t.grad.data

        def calc_grad(x):
            var_t = nn.Parameter(self.image_model.transform(img).unsqueeze(0))
            feat_t = self.get_image_features(var_t)
        
            feat_t.backward(torch.tensor(x).view(1, n_features), retain_graph=True)
            return var_t.grad

        if rank > max_items:

            y_uit = self.model.score_user_item(u_i, t, feat_t).item()
            rank_u_i = self.get_rank(u_i_scores, y_uit)

            y_ut_tag = self.model.score_user_item(u_tag, t, feat_t).item()
            rank_u_tag = self.get_rank(u_tag_scores, y_ut_tag)

            print(f"Ranks: {rank}, {rank_u_i}, {rank_u_tag}")

            if rank_u_tag <= max_items:
            
                W_i, y_i = self.build_eq_attach(
                    u_i, t, img_M, feat_t, rank_u_i, u_i_scores, n_examples, n_features, gamma
                )
                W_tag, y_tag = self.build_eq_attach(
                    u_tag, t, img_M, feat_t, rank_u_tag, u_tag_scores, n_examples, n_features, gamma
                )

                x_itag = solve(W_i, y_i)
                x_utag = solve(W_tag, y_tag)

                x_apx = ((n+1)*x_utag - x_itag)/n

                grad = calc_grad(x_apx)
            
            else:
                W_i, y_i = self.build_eq_attach(
                    u_i, t, img_M, feat_t, rank_u_i, u_i_scores, n_examples, n_features, gamma
                )
                x_itag = solve(W_i, y_i)
                grad = calc_grad(x_itag)

        else:
            print(f"Rank: {rank}")

            W, y = self.build_eq_attach(
                u, t, img_M, feat_t, rank, u_scores, n_examples, n_features, gamma
            )

            x = solve(W, y)
            grad = calc_grad(x)

        grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
        img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)

        return img_M, rank

    def bb_attack_i_itag(
            self, u, u_i, u_tag, t, img, n, n_examples, n_features, epsilon, gamma
        ):

        max_items = len(self.items_data)//100
        
        img_M = np.array(img)
        var_t = nn.Parameter(self.image_model.transform(img).unsqueeze(0))
        feat_t = self.get_image_features(var_t)

        def get_u_scores(u):
            u_scores = self.model.score(u).detach().numpy().squeeze()
            u_scores.sort()
            return np.flip(u_scores)

        u_scores = get_u_scores(u)
        u_i_scores = get_u_scores(u_i)
        u_tag_scores = get_u_scores(u_tag)

        y_ut = self.model.score_user_item(u, t, feat_t.detach()).item()
        rank = self.get_rank(u_scores, y_ut)

        y_uit = self.model.score_user_item(u_i, t, feat_t.detach()).item()
        rank_u_i = self.get_rank(u_i_scores, y_uit)

        y_ut_tag = self.model.score_user_item(u_tag, t, feat_t.detach()).item()
        rank_u_tag = self.get_rank(u_tag_scores, y_ut_tag)

        print(f"Ranks: {rank}, {rank_u_i}, {rank_u_tag}")

        if rank_u_tag > max_items:

            W, y = self.build_eq_attach(
                u_i, t, img_M, feat_t, rank_u_i, u_i_scores, n_examples, n_features, gamma
            )

            x = solve(W, y)
            
            def attack(x, feat_t, var_t, img_M):
                feat_t.backward(torch.tensor(x).view(1, n_features), retain_graph=True)

                grad = var_t.grad

                grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
                img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)

                return img_M

            img_M = attack(x, feat_t, var_t, img_M)

        return img_M, rank, rank_u_i, rank_u_tag

    def search_for_high_ranked_item(self, similar_items, t):

        max_items = len(self.items_data)//100

        from itertools import combinations

        img = self.load_image(t)

        var_t = nn.Parameter(self.image_model.transform(img).unsqueeze(0))
        feat_t = self.get_image_features(var_t)

        def get_u_scores(u):
            u_scores = self.model.score(u).detach().numpy().squeeze()
            u_scores.sort()
            return np.flip(u_scores)

        for i in similar_items:
            u = self.model.add_fake_user_by_item(i)
            u_scores = get_u_scores(u)
            y_u = self.model.score_user_item(u, t, feat_t.detach()).item()
            rank = self.get_rank(u_scores, y_u)
            if rank < max_items:
                return i, rank
        return None, None


    def run_bb_single_user_restricted_items(self, args, logger):

        # max_items = len(self.items_data)//100

        steps = args.steps
        n_examples = args.examples
        epsilon = int(args.epsilon * 255)
        gamma = args.gamma
        # i_tag = args.i_tag

        u = args.user
        i = args.item

        self.run_bb_restricted_attack(u, i, steps, n_examples, epsilon, gamma, logger)


    def run_bb_restricted_attack(self, u, i, steps, n_examples, epsilon, gamma, logger):

        max_items = len(self.items_data)//100

        print("Start:", u, i)

        scores = self.model.score_similar_items(i)
        similar_items = scores.argsort(dim=0, descending=True).squeeze().numpy()
        for i_tag in similar_items[:10]:

            user_items = self.user_items[u]
            u_tag = self.model.add_fake_user_by_other_and_add_items(u, user_items, [i_tag])
            u_i_tag = self.model.add_fake_user_by_item(i_tag)

            n_features = self.model.F.shape[1]

            img = self.load_image(i)

            print("phase 1")

            step = 1

            prev_ranks = (None, None, None)

            while step <= steps:
                img_M, rank_u, rank_u_i, rank_u_tag = self.bb_attack_i_itag(
                    u, u_i_tag, u_tag, i, img, len(user_items), n_examples, n_features, epsilon, gamma
                )
                img = Image.fromarray(img_M)

                if prev_ranks == (rank_u, rank_u_i, rank_u_tag):
                    break

                if rank_u_tag <= max_items:
                    break

                prev_ranks = (rank_u, rank_u_i, rank_u_tag)

                step += 1

            if rank_u_tag > max_items:
                print("Trying next i_tag")
                continue

            print("phase 2")

            while step <= steps:
                img_M, rank = self.bb_attack_limitted_returned_items(
                    u, u_i_tag, u_tag, i, img, len(user_items), n_examples, n_features, epsilon, gamma
                )
                img = Image.fromarray(img_M)

                if rank <= 1:
                    break

                step += 1

            if rank <= 1:
                print("Success!")
                break


    def run_restrict_attack_rate(self, args, logger):
        max_items = len(self.items_data)//100
        repeat = 100000
        random_users = random.sample(range(exp.model.n_users), k=repeat)
        random_items = random.sample(range(exp.model.n_items), k=repeat)
        checked = set()
        total, successes = 0, 0
        for u, i in tqdm(zip(random_users, random_items), total=repeat):
            if (u,i) in checked:
                continue
            checked.add((u,i))
            u_i = self.model.add_fake_user_by_item(i)

            scores = self.model.score_items_user(u_i)
            similar_items = scores.argsort(dim=0, descending=True).squeeze().numpy()
            i_tag, rank = self.search_for_high_ranked_item(similar_items[:max_items], i)

            total += 1

            if i_tag is not None:
                successes += 1
                print(f"{total}, {successes}, {u}, {i}, {i_tag}, {rank}")

        
    def run_bb_single_user(self, args, logger):
        u = args.user
        do_pca = args.do_pca
        by_rank = args.by_rank
        rank_distribution = "uniform"
        if by_rank:
            rank_distribution = args.rank_distribution
        steps = args.steps
        epsilon = int(args.epsilon * 255)
        gamma = args.gamma
        n_examples = args.examples

        u_scores = self.model.score(u).detach().numpy().squeeze()
        u_scores.sort()
        u_scores = np.flip(u_scores)

        if hasattr(args, 'item') and args.item is not None:
            t = args.item
            result = self.search(u, -1)
            from_rank = np.where(result==t)[0][0].item() + 1
        else:
            from_rank = args.from_rank
            result = self.search(u, from_rank)
            t = result[from_rank-1].item()

        print(f"Run single user, black-box, for user {u} from rank {from_rank}")

        logger.log({
            'user': u,
            'item': t,
            'rank': from_rank,
        })

        backup = self.model.F[t].unsqueeze(0).clone().detach()

        n_features = backup.shape[1]

        pca_fc = None
        if do_pca:
            n_components = args.n_components
            pca_fc = TorchPCA.get_pca(self.model.F.numpy(), n_components)
            n_features = n_components
        
        img = self.load_image(t)
        img_M = np.array(img)

        if not os.path.exists(f"images/{t}"):
            os.makedirs(f"images/{t}")
        img.save(f"images/{t}/original.jpeg", "JPEG")

        ori_img_M = img_M
        eps = 10

        for step in range(steps):

            img_M = self.bb_attack(
                u, t, img, img_M, ori_img_M, n_examples, n_features, u_scores, 
                do_pca, pca_fc, by_rank, rank_distribution, epsilon, eps, gamma
            )

            img = Image.fromarray(img_M)

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))
            score = self.model.score_user_item(u, t, feat_t.detach()).item()
            rank = self.get_rank(u_scores, score)

            print(f"Step: {step+1}, rank={rank}")

            logger.log({
                'step': step,
                'rank': rank,
            })

            img.save(f"images/{t}/step_{step}.jpeg", "JPEG")

            if rank == 1:
                break
    
        return img

    def run_bb_multi_users(self, args, logger):
        users = random.sample(range(self.model.n_users), k=10)
        t = random.choice(range(self.model.n_items))
        steps = args.steps
        epsilon = int(args.epsilon * 255)
        gamma = args.gamma
        n_examples = args.examples

        backup = self.model.F[t].unsqueeze(0).clone().detach()

        n_features = backup.shape[1]

        users_scores = {}
        for u in users:
            u_scores = self.model.score(u).detach().numpy().squeeze()
            u_scores.sort()
            u_scores = np.flip(u_scores)
            users_scores[u] = u_scores

        img = self.load_image(t)
        img_M = np.array(img)

        for step in range(steps):
            print(f"Step: {step+1}")

            for u in users:
                W, y = [], []

                img_t = self.image_model.transform(img)
                img_t = img_t.unsqueeze(0)
                var_t = nn.Parameter(img_t)
                feat_t = self.get_image_features(var_t)

                y_t = self.model.score_user_item(u, t, feat_t.detach()).item()

                rank = self.get_rank(u_scores, y_t)
                print(f"Step: {step+1}, user={u}, rank={rank}")

                for i in tqdm(range(n_examples)):#range(n_features)):
                    d = np.random.choice(range(-gamma, gamma+1), size=img_M.shape)
                    img_i = np.clip(img_M.astype(np.int) + d, 0, 255).astype(np.uint8)
                    img_i = Image.fromarray(img_i)

                    img_i = self.image_model.transform(img_i)
                    feat_i = self.get_image_features(img_i.unsqueeze(0))

                    y_i = self.model.score_user_item(u, t, feat_i).item()

                    diff = feat_i-feat_t

                    W.append(diff.detach().numpy())
                    y.append(y_t-y_i)

                x = solve(np.vstack(W), y)
                
                feat_t.backward(torch.tensor(x).view(1, n_features), retain_graph=True)

                grad = var_t.grad
                
                grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
                img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)

                img = Image.fromarray(img_M)

                logger.log({
                    'step': step,
                    'rank': rank,
                })

        for u in users:
            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))
            y_t = self.model.score_user_item(u, t, feat_t.detach()).item()
            rank = self.get_rank(u_scores, y_t)

            logger.log({
                'step': step,
                'rank': rank,
            })

            print(f"Rank: {rank}")
    
        return img

    def get_worst_user(self, users, t, t_feat, users_scores):
        max_rank, max_user = 0, -1
        for u in users:
            u_scores = users_scores[u]
            score = self.model.score_user_item(u, t, t_feat).item()
            rank = self.get_rank(u_scores, score)
            if rank > max_rank:
                max_rank, max_user = rank, u
        return max_user

    
    def get_median_user(self, users, t, t_feat, users_scores, percentile=0.25):
        ranks = []
        d = int(1 / percentile)
        for u in users:
            u_scores = users_scores[u]
            score = self.model.score_user_item(u, t, t_feat).item()
            rank = self.get_rank(u_scores, score)
            ranks.append(rank)
        i = np.argsort(ranks)[len(ranks)//d]
        return users[i]


    def run_wb_multi_users(self, args, logger):
        users = random.sample(range(self.model.n_users), k=10)
        t = random.choice(range(self.model.n_items))
        steps = args.steps
        epsilon = args.epsilon

        users_scores = {}
        for u in users:
            u_scores = self.model.score(u).detach().numpy().squeeze()
            u_scores.sort()
            u_scores = np.flip(u_scores)
            users_scores[u] = u_scores

        t_img = self.load_image(t)

        t_img_tran = self.image_model.transform(t_img)
        t_img_tran = torch.unsqueeze(t_img_tran, 0)
        t_img_var = nn.Parameter(t_img_tran)
        t_feat = self.get_image_features(t_img_var)

        for step in range(steps):
            print(f"Step: {step+1}")
            for u in users:
                u_scores = users_scores[u]

                score = self.model.score_user_item(u, t, t_feat)
                init_rank = self.get_rank(u_scores, score.detach())
                if init_rank == 1:
                    continue
                
                zero_gradients(t_img_var)
                loss = self.model.pointwise_forward([u], [t], t_feat)
                loss.backward(retain_graph=True)

                grad = torch.sign(t_img_var.grad.data)
                adversarial = t_img_var.data - epsilon * grad
                t_img_var.data = adversarial
                t_feat = self.get_image_features(t_img_var)

                score = self.model.score_user_item(u, t, t_feat)
                rank = self.get_rank(u_scores, score.detach())
                print(f"\t{u}, {init_rank}->{rank}")

                logger.log({
                    'step': step,
                    'rank': rank,
                    'user': u,
                })

        print(f"Final ranks:")
        for u in users:
            u_scores = users_scores[u]
            score = -self.model.pointwise_forward([u], [t], t_feat).item()
            rank = self.get_rank(u_scores, score)

            print(f"\t{u}, {rank}")

            logger.log({
                'step': step,
                'rank': rank,
                'user': u,
            })

        return Image.fromarray(to_img(adversarial))


    def run_bb_multi_users_m2(self, args, logger):
        users = random.sample(range(self.model.n_users), k=100)
        t = random.choice(range(self.model.n_items))
        steps = args.steps
        epsilon = int(args.epsilon * 255)
        gamma = args.gamma
        n_examples = args.examples

        feat_t = self.model.F[t].unsqueeze(0).clone().detach()

        n_features = feat_t.shape[1]

        users_scores = {}
        for u in users:
            u_scores = self.model.score(u).detach().numpy().squeeze()
            u_scores.sort()
            u_scores = np.flip(u_scores)
            users_scores[u] = u_scores

        img = self.load_image(t)
        img_M = np.array(img)

        ori_img_M = img_M

        for step in range(steps):
            W, y = [], []

            img_t = self.image_model.transform(img)
            var_t = nn.Parameter(img_t.unsqueeze(0))
            feat_t = self.get_image_features(var_t)

            u = self.get_median_user(users, t, feat_t.detach(), users_scores)
            u_scores = users_scores[u]
            y_t = self.model.score_user_item(u, t, feat_t.detach()).item()

            init_rank = self.get_rank(u_scores, y_t)

            for _ in range(n_examples):
                d = np.random.choice(range(-gamma, gamma+1), size=img_M.shape)
                img_i = np.clip(img_M.astype(np.int) + d, 0, 255).astype(np.uint8)
                img_i = Image.fromarray(img_i)

                img_i = self.image_model.transform(img_i)
                feat_i = self.get_image_features(img_i.unsqueeze(0))

                y_i = self.model.score_user_item(u, t, feat_i).item()

                diff = feat_i-feat_t

                W.append(diff.detach().numpy())
                y.append(y_t-y_i)

            x = solve(np.vstack(W), y)
            
            feat_t.backward(torch.tensor(x).view(1, n_features), retain_graph=True)

            grad = var_t.grad

            attack = "ifgsm"
            eps = 20

            # fgsm, ifgsm, pgd, first
            if attack == "first":
                grad = grad.squeeze(0).numpy().transpose((1, 2, 0))
                grad /= norm(grad)
                grad = np.clip(grad*255, -epsilon, epsilon)
                img_M = np.clip(img_M.astype(np.int) - grad, 0, 255).astype(np.uint8)
            elif attack == "ifgsm":
                grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
                img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)
            elif attack == "pgd":
                grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
                img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255)
                eta = np.clip(img_M - ori_img_M, -eps, eps)
                img_M = np.clip(ori_img_M+eta, 0, 255).astype(np.uint8)
            
            # grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
            # img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_M)

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))
            score = self.model.score_user_item(u, t, feat_t.detach()).item()
            rank = self.get_rank(u_scores, score)

            print(f"Step: {step+1}, user={u}, {init_rank}->{rank}")

            logger.log({
                'step': step,
                'rank': rank,
            })

        for u in users:
            u_scores = users_scores[u]
            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))
            score = self.model.score_user_item(u, t, feat_t.detach()).item()
            rank = self.get_rank(u_scores, score)

            logger.log({
                'step': step,
                'rank': rank,
            })

            print(f"Rank: {rank}")
    
        return img

    def run_wb_multi_users_m2(self, args, logger):
        users = random.sample(range(self.model.n_users), k=10)
        t = random.choice(range(self.model.n_items))
        steps = args.steps
        epsilon = args.epsilon

        t_img = self.load_image(t)

        t_img_tran = self.image_model.transform(t_img)
        t_img_tran = torch.unsqueeze(t_img_tran, 0)
        t_img_var = nn.Parameter(t_img_tran)
        t_feat = self.get_image_features(t_img_var)

        users_scores = {}
        for u in users:
            u_scores = self.model.score(u).detach().numpy().squeeze()
            u_scores.sort()
            u_scores = np.flip(u_scores)
            users_scores[u] = u_scores

            score = self.model.score_user_item(u, t, t_feat).item()
            rank = self.get_rank(u_scores, score)

            print(u, rank, score)

        for step in range(steps):
            u = self.get_worst_user(users, t, t_feat.detach(), users_scores)
            u_scores = users_scores[u]

            score = self.model.score_user_item(u, t, t_feat).item()
            init_rank = self.get_rank(u_scores, score)
            
            zero_gradients(t_img_var)
            loss = self.model.pointwise_forward([u], [t], t_feat)
            loss.backward(retain_graph=True)

            grad = torch.sign(t_img_var.grad.data)
            adversarial = t_img_var.data - epsilon * grad
            t_img_var.data = adversarial
            t_feat = self.get_image_features(t_img_var)

            score = self.model.score_user_item(u, t, t_feat).item()
            rank = self.get_rank(u_scores, score)
            print(f"Step: {step+1}, user={u}, {init_rank}->{rank}")

            logger.log({
                'step': step,
                'rank': rank,
                'user': u,
            })

        print(f"Final ranks:")
        for u in users:
            u_scores = users_scores[u]
            score = -self.model.pointwise_forward([u], [t], t_feat).item()
            rank = self.get_rank(u_scores, score)

            print(f"\t{u}, {rank}")

            logger.log({
                'step': step,
                'rank': rank,
                'user': u,
            })

        return Image.fromarray(to_img(adversarial))


    def run_wb_general_pop(self, args, logger):

        random_users = random.sample(range(self.model.n_users), k=1000)
        t = args.item
        steps = args.steps
        epsilon = args.epsilon

        fake_users = []
        random_items = random.sample(range(self.model.n_items), k=200)
        for item in random_items:
            user = self.model.add_fake_user_by_item(item)
            fake_users.append(user)

        img = self.load_image(t)
        img_M = np.array(img)

        t_img_tran = self.image_model.transform(img)
        t_img_tran = torch.unsqueeze(t_img_tran, 0)
        t_img_var = nn.Parameter(t_img_tran)
        t_feat = self.get_image_features(t_img_var)

        users_scores = {}
        for us in [fake_users, random_users]:
            scores = self.model.score_users(us).detach().numpy()
            item_scores = self.model.score_users_item(us, t).detach().numpy()
            scores.sort(axis=-1)
            scores = np.flip(scores, axis=-1)
            ranks = []
            for u, u_scores, score in zip(us, scores, item_scores):
                users_scores[u] = u_scores
                rank = self.get_rank(u_scores, score)
                ranks.append(rank)

            logger.log({
                'step': 0,
                'rank': ranks,
            })
            ranks = np.array(ranks)

            print((ranks <= 20).mean())

        for step in range(steps):
            u = self.get_median_user(fake_users, t, t_feat.detach(), users_scores)
            u_scores = users_scores[u]

            score = self.model.score_user_item(u, t, t_feat).item()
            init_rank = self.get_rank(u_scores, score)

            t_img_tran = self.image_model.transform(img)
            t_img_tran = torch.unsqueeze(t_img_tran, 0)
            t_img_var = nn.Parameter(t_img_tran)
            t_feat = self.get_image_features(t_img_var)
            
            zero_gradients(t_img_var)
            loss = self.model.pointwise_forward([u], [t], t_feat)
            loss.backward(retain_graph=True)

            grad = t_img_var.grad.data
            grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
            img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_M)

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))
            score = self.model.score_user_item(u, t, feat_t.detach()).item()
            rank = self.get_rank(u_scores, score)

            print(f"Step: {step+1}, user={u}, {init_rank}->{rank}")

            for us in [fake_users, random_users]:
                ranks = []
                for u in us:
                    u_scores = users_scores[u]
                    score = -self.model.pointwise_forward([u], [t], t_feat).item()
                    rank = self.get_rank(u_scores, score)

                    ranks.append(rank)

                logger.log({
                    'step': step+1,
                    'rank': ranks,
                })

                ranks = np.array(ranks)
                print((ranks <= 20).mean())

        print(f"Final ranks:")
        for u in fake_users:
            u_scores = users_scores[u]
            score = -self.model.pointwise_forward([u], [t], t_feat).item()
            rank = self.get_rank(u_scores, score)

            print(f"\t{u}, {rank}")

            logger.log({
                'step': step,
                'rank': rank,
                'user': u,
            })

        return img


    def run_bb_general_pop(self, args, logger):

        random_users = random.sample(range(self.model.n_users), k=1000)
        t = args.item
        steps = args.steps
        epsilon = args.epsilon
        gamma = args.gamma
        n_examples = args.examples

        fake_users = []
        random_items = random.sample(range(self.model.n_items), k=200)
        for item in random_items:
            user = self.model.add_fake_user_by_item(item)
            fake_users.append(user)

        img = self.load_image(t)
        img_M = np.array(img)

        ori_img_M = img_M

        t_img_tran = self.image_model.transform(img)
        t_img_tran = torch.unsqueeze(t_img_tran, 0)
        t_img_var = nn.Parameter(t_img_tran)
        t_feat = self.get_image_features(t_img_var)

        n_features = t_feat.shape[1]

        users_scores = {}
        for us in [fake_users, random_users]:
            scores = self.model.score_users(us).detach().numpy()
            item_scores = self.model.score_users_item(us, t).detach().numpy()
            scores.sort(axis=-1)
            scores = np.flip(scores, axis=-1)
            ranks = []
            for u, u_scores, score in zip(us, scores, item_scores):
                users_scores[u] = u_scores
                rank = self.get_rank(u_scores, score)
                ranks.append(rank)

            logger.log({
                'step': 0,
                'rank': ranks,
            })
            ranks = np.array(ranks)

            print((ranks <= 20).mean())

        for step in range(steps):
            W, y = [], []

            img_t = self.image_model.transform(img)
            var_t = nn.Parameter(img_t.unsqueeze(0))
            feat_t = self.get_image_features(var_t)

            u = self.get_median_user(fake_users, t, feat_t.detach(), users_scores)
            u_scores = users_scores[u]
            y_t = self.model.score_user_item(u, t, feat_t.detach()).item()

            init_rank = self.get_rank(u_scores, y_t)

            for _ in range(n_examples):
                d = np.random.choice(range(-gamma, gamma+1), size=img_M.shape)
                img_i = np.clip(img_M.astype(np.int) + d, 0, 255).astype(np.uint8)
                img_i = Image.fromarray(img_i)

                img_i = self.image_model.transform(img_i)
                feat_i = self.get_image_features(img_i.unsqueeze(0))

                y_i = self.model.score_user_item(u, t, feat_i).item()

                diff = feat_i-feat_t

                W.append(diff.detach().numpy())
                y.append(y_t-y_i)

            x = solve(np.vstack(W), y)
            
            feat_t.backward(torch.tensor(x).view(1, n_features), retain_graph=True)

            grad = var_t.grad
            
            grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
            img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_M)

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))
            score = self.model.score_user_item(u, t, feat_t.detach()).item()
            rank = self.get_rank(u_scores, score)

            print(f"Step: {step+1}, user={u}, {init_rank}->{rank}")

            for us in [fake_users, random_users]:
                ranks = []
                for u in us:
                    u_scores = users_scores[u]
                    score = -self.model.pointwise_forward([u], [t], feat_t).item()
                    rank = self.get_rank(u_scores, score)

                    ranks.append(rank)

                logger.log({
                    'step': step+1,
                    'rank': ranks,
                })

                ranks = np.array(ranks)
                print((ranks <= 20).mean())

        print(f"Final ranks:")
        for u in fake_users:
            u_scores = users_scores[u]
            score = -self.model.pointwise_forward([u], [t], feat_t).item()
            rank = self.get_rank(u_scores, score)

            print(f"\t{u}, {rank}")

            logger.log({
                'step': step,
                'rank': rank,
                'user': u,
            })

        return img

    def search_i_tag(self, similar_items, i, img):

        def get_u_scores(u):
            u_scores = self.model.score(u).detach().numpy().squeeze()
            u_scores.sort()
            return np.flip(u_scores)

        max_items = len(self.items_data)//100

        feat_t = self.get_image_features(self.image_model.transform(img).unsqueeze(0))

        for i_tag in similar_items[:25]:
            u_i_tag = self.model.add_fake_user_by_item(i_tag)

            u_i_scores = get_u_scores(u_i_tag)

            y_uit = self.model.score_user_item(u_i_tag, i, feat_t).item()
            rank_u_i = self.get_rank(u_i_scores, y_uit)

            if rank_u_i <= max_items:
                return i_tag, u_i_tag

        return None, None

    def run_bb_restricted_attack_2(self, u, i, steps, n_examples, epsilon, gamma):

        img = self.load_image(i)

        scores = self.model.score_similar_items(i)
        similar_items = scores.argsort(dim=0, descending=True).squeeze().numpy()
        i_tag, u_i_tag = self.search_i_tag(similar_items, i, img)

        if i_tag is not None:

            u_tag = self.model.add_fake_user_by_other_and_add_items(u, [i], [i_tag])

            n_features = self.model.F.shape[1]

            step = 1
            while step <= steps:
                img_M, rank = self.bb_attack_limitted_returned_items(
                    u, u_i_tag, u_tag, i, img, 1, n_examples, n_features, epsilon, gamma
                )
                img = Image.fromarray(img_M)
                step += 1
                yield step, rank, img


    def run_bb_general_restricted_attack(self, args, logger):

        random_users = random.sample(range(self.model.n_users), k=1000)
        t = args.item
        steps = args.steps
        epsilon = args.epsilon
        gamma = args.gamma
        n_examples = args.examples

        fake_users = []
        random_items = random.sample(range(self.model.n_items), k=200)
        for item in random_items:
            user = self.model.add_fake_user_by_item(item)
            fake_users.append(user)

        img = self.load_image(t)

        users_scores = {}
        for us in [fake_users, random_users]:
            scores = self.model.score_users(us).detach().numpy()
            item_scores = self.model.score_users_item(us, t).detach().numpy()
            scores.sort(axis=-1)
            scores = np.flip(scores, axis=-1)
            ranks = []
            for u, u_scores, score in zip(us, scores, item_scores):
                users_scores[u] = u_scores
                rank = self.get_rank(u_scores, score)
                ranks.append(rank)

            logger.log({
                'step': 0,
                'rank': ranks,
            })
            ranks = np.array(ranks)

            print((ranks <= 20).mean())

        feat_t = self.get_image_features(self.image_model.transform(img).unsqueeze(0))
        u = self.get_median_user(fake_users, t, feat_t.detach(), users_scores, 0.25)
        
        patient = 12
        prev_rank = -1

        for step, rank, img in self.run_bb_restricted_attack_2(u, t, steps, n_examples, epsilon, gamma):

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))
            score = self.model.score_user_item(u, t, feat_t.detach()).item()
            rank = self.get_rank(u_scores, score)

            print(f"Step: {step}, user={u}, {rank}")

            for us in [fake_users, random_users]:
                ranks = []
                for u in us:
                    u_scores = users_scores[u]
                    score = -self.model.pointwise_forward([u], [t], feat_t).item()
                    r = self.get_rank(u_scores, score)

                    ranks.append(r)

                logger.log({
                    'step': step,
                    'rank': ranks,
                })

                ranks = np.array(ranks)
                print((ranks <= 20).mean())

            if rank == prev_rank:
                patient -= 1
                if patient == 0:
                    break
            else:
                prev_rank = rank
                patient = 12
            step += 1

            u = self.get_median_user(fake_users, t, feat_t.detach(), users_scores)

        # print(f"Final ranks:")
        # for u in fake_users:
        #     u_scores = users_scores[u]
        #     score = -self.model.pointwise_forward([u], [t], feat_t).item()
        #     rank = self.get_rank(u_scores, score)

        #     print(f"\t{u}, {rank}")

        #     logger.log({
        #         'step': step,
        #         'rank': rank,
        #         'user': u,
        #     })

        return img


    def run_bb_segment_attack(self, args, logger):
        do_pca = args.do_pca
        by_rank = args.by_rank
        if by_rank:
            rank_distribution = args.rank_distribution
        steps = args.steps
        epsilon = int(args.epsilon * 255)
        gamma = args.gamma
        n_examples = args.examples
        t = i = args.item

        while True:
            users = [u for u, items in enumerate(self.user_items) if t in items]
            if len(users) >= 10:
                break
            t = (t+1) % len(self.items_data)

        print(len(users))

        random_users = random.sample(range(self.model.n_users), k=100)

        args.user = user = self.model.add_fake_user_by_item(i)

        user_scores = self.model.score(user).detach().numpy().squeeze()
        user_scores.sort()
        user_scores = np.flip(user_scores)

        print(f"Segment bb experiment for user={user}, item={t}")
        logger.log({
            'user': user,
            'target_item': t,
            'seed_item': i,
        })

        users_scores = {}
        for us in [users, random_users]:
            scores = self.model.score_users(us).detach().numpy()
            item_scores = self.model.score_users_item(us, t).detach().numpy()
            scores.sort(axis=-1)
            scores = np.flip(scores, axis=-1)
            ranks = []
            for u, u_scores, score in zip(us, scores, item_scores):
                users_scores[u] = u_scores
                rank = self.get_rank(u_scores, score)
                ranks.append(rank)

            logger.log({
                'step': 0,
                'rank': ranks,
            })
            ranks = np.array(ranks)

            print((ranks <= 20).mean())

        backup = self.model.F[t].unsqueeze(0).clone().detach()

        n_features = backup.shape[1]

        pca_fc = None
        if do_pca:
            n_components = args.n_components
            pca_fc = TorchPCA.get_pca(self.model.F.numpy(), n_components)
            n_features = n_components

        img = self.load_image(t)
        img_M = np.array(img)
        ori_img_M = img_M

        eps = 30

        for step in range(steps):
            print(f"Step: {step}")

            img_M = self.bb_attack(
                user, t, img, img_M, ori_img_M, n_examples, n_features, user_scores, 
                do_pca, pca_fc, by_rank, rank_distribution, epsilon, eps, gamma
            )

            img = Image.fromarray(img_M)

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0)).detach()

            for us in [users, random_users]:
                # scores = self.model.score_users_item(us, t, feat_t)
                ranks = []
                for u in us:
                    u_scores = users_scores[u]
                    score = -self.model.pointwise_forward([u], [t], feat_t).item()
                    rank = self.get_rank(u_scores, score)

                    ranks.append(rank)

                logger.log({
                    'step': step+1,
                    'rank': ranks,
                })
                ranks = np.array(ranks)
                print((ranks <= 20).mean())

        return img


    def is_bb_restricted_attack_possible(self, u, i, steps, n_examples, epsilon, gamma):

        max_items = len(self.items_data)//100

        scores = self.model.score_similar_items(i)
        similar_items = scores.argsort(dim=0, descending=True).squeeze().numpy()
        for i_tag in similar_items[:3]:
            u_tag = self.model.add_fake_user_by_other_and_add_items(u, [i], [i_tag])
            u_i_tag = self.model.add_fake_user_by_item(i_tag)

            n_features = self.model.F.shape[1]

            img = self.load_image(i)

            step = 1

            prev_ranks = (None, None, None)

            while step <= steps:
                img_M, rank_u, rank_u_i, rank_u_tag = self.bb_attack_i_itag(
                    u, u_i_tag, u_tag, i, img, 1, n_examples, n_features, epsilon, gamma
                )
                img = Image.fromarray(img_M)

                if prev_ranks == (rank_u, rank_u_i, rank_u_tag):
                    break

                if rank_u_tag <= max_items:
                    break

                prev_ranks = (rank_u, rank_u_i, rank_u_tag)

                step += 1

            if rank_u_tag > max_items:
                continue

            phase_1_steps = step

            patient = 3
            prev_rank = 1000000
            while step <= steps:
                img_M, rank = self.bb_attack_limitted_returned_items(
                    u, u_i_tag, u_tag, i, img, 1, n_examples, n_features, epsilon, gamma
                )
                img = Image.fromarray(img_M)

                if rank >= prev_rank:
                    patient -= 1
                    if patient == 0:
                        break
                else:
                    patient = 3

                if rank <= 20:
                    return img_M, phase_1_steps, u_i_tag, u_tag

                step += 1
                prev_rank = rank

        return None, -1, -1, -1


    def run_bb_segment_restricted_attack(self, args, logger):
        steps = args.steps
        epsilon = int(args.epsilon * 255)
        gamma = args.gamma
        n_examples = args.examples
        i = random.randint(0, len(self.items_data)-1) # push
        t = random.randint(0, len(self.items_data)-1) # segment

        while True:
            users = [u for u, items in enumerate(self.user_items) if t in items]
            if len(users) >= 10:
                break
            t = (t+1) % len(self.items_data)

        print(len(users))

        random_users = random.sample(range(self.model.n_users), k=10)

        args.user = user = self.model.add_fake_user_by_item(t)

        user_scores = self.model.score(user).detach().numpy().squeeze()
        user_scores.sort()
        user_scores = np.flip(user_scores)

        print(f"Segment bb experiment for user={user}, push item={i}, segment item={t}")
        logger.log({
            'user': user,
            'push_item': i,
            'segment_item': t,
        })

        users_scores = {}
        for us in [users, random_users]:
            scores = self.model.score_users(us).detach().numpy()
            item_scores = self.model.score_users_item(us, i).detach().numpy()
            scores.sort(axis=-1)
            scores = np.flip(scores, axis=-1)
            ranks = []
            for u, u_scores, score in zip(us, scores, item_scores):
                users_scores[u] = u_scores
                rank = self.get_rank(u_scores, score)
                ranks.append(rank)

            logger.log({
                'step': 0,
                'rank': ranks,
            })
            ranks = np.array(ranks)

            print((ranks <= 20).mean())

        img = None
        patient = 5
        prev_rank = -1

        for step, rank, img in self.run_bb_restricted_attack_2(user, i, steps, n_examples, epsilon, gamma):
            
            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0)).detach()

            for us in [users, random_users]:
                ranks = []
                for u in us:
                    u_scores = users_scores[u]
                    score = -self.model.pointwise_forward([u], [t], feat_t).item()
                    r = self.get_rank(u_scores, score)

                    ranks.append(r)

                logger.log({
                    'step': step,
                    'rank': ranks,
                })
                ranks = np.array(ranks)
                print((ranks <= 20).mean())

            if rank == prev_rank:
                patient -= 1
                if patient == 0:
                    break
            else:
                prev_rank = rank
                patient = 5
            step += 1

        return img

    def run_wb_segment_attack(self, args, logger):

        steps = args.steps
        epsilon = args.epsilon
        t = i = args.item

        while True:
            users = [u for u, items in enumerate(self.user_items) if t in items]
            if len(users) >= 10:
                break
            t = (t+1) % len(self.items_data)

        print(len(users))

        random_users = random.sample(range(self.model.n_users), k=100)

        args.user = user = self.model.add_fake_user_by_item(i)
        
        # t = random.choice(range(self.model.n_items))

        print(f"Segment wb experiment for user={user}, item={t}")
        logger.log({
            'user': user,
            'target_item': t,
            'seed_item': i,
        })

        users_scores = {}
        for us in [users, random_users]:
            scores = self.model.score_users(us).detach().numpy()
            item_scores = self.model.score_users_item(us, t).detach().numpy()
            scores.sort(axis=-1)
            scores = np.flip(scores, axis=-1)
            ranks = []
            for u, u_scores, score in zip(us, scores, item_scores):
                users_scores[u] = u_scores
                rank = self.get_rank(u_scores, score)
                ranks.append(rank)

            logger.log({
                'step': 0,
                'rank': ranks,
            })
            ranks = np.array(ranks)

            print((ranks <= 20).mean())

        img = self.load_image(t)
        img_M = np.array(img)

        for step in range(steps):
            print(f"Step: {step}")

            t_img_tran = self.image_model.transform(img)
            t_img_tran = torch.unsqueeze(t_img_tran, 0)
            t_img_var = nn.Parameter(t_img_tran)
            t_feat = self.get_image_features(t_img_var)

            zero_gradients(t_img_var)
            loss = self.model.pointwise_forward([user], [t], t_feat)
            loss.backward(retain_graph=True)

            grad = t_img_var.grad.data
            grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
            img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_M)

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))

            for us in [users, random_users]:
                ranks = []
                for u in us:
                    u_scores = users_scores[u]
                    score = -self.model.pointwise_forward([u], [t], feat_t).item()
                    rank = self.get_rank(u_scores, score)

                    ranks.append(rank)

                logger.log({
                    'step': step+1,
                    'rank': ranks,
                })

                ranks = np.array(ranks)
                # print(ranks)
                # print(ranks.mean())
                print((ranks <= 20).mean())

        return img


    def run_baseline(self, args, logger):
        u = args.user

        u_scores = self.model.score(u).detach().numpy().squeeze()
        u_scores.sort()
        u_scores = np.flip(u_scores)

        if hasattr(args, 'item') and args.item is not None:
            t = args.item
            result = self.search(u, -1)
            from_rank = np.where(result==t)[0][0].item() + 1
        else:
            from_rank = args.from_rank
            result = self.search(u, from_rank)
            t = result[from_rank-1].item()

        top = self.search(u, 1)[0].item()

        t_img = self.load_image(t)
        top_img = self.load_image(top)

        print(f"Run single user baseline, for user {u} from rank {from_rank}")

        logger.log({
            'user': u,
            'item': t,
            'rank': from_rank,
        })

        img_M = np.array(t_img)
        top_M = np.array(top_img)

        img_M = np.clip(img_M.astype(np.float) - 0.07*top_M.astype(np.float), 0, 255).astype(np.uint8)

        img = Image.fromarray(img_M)

        img_t = self.image_model.transform(img)
        feat_t = self.get_image_features(img_t.unsqueeze(0))
        score = self.model.score_user_item(u, t, feat_t.detach()).item()
        rank = self.get_rank(u_scores, score)

        print(f"Rank: {rank}")

        logger.log({
            'step': 0,
            'rank': rank,
        })

        return img


    def run(self, name, args):

        # ts = time.strftime("%y%m%d-%H%M%S")
        logfile = f"{name}.log"

        with JsonLooger(logfile) as logger:
            logger.log({k:v for k,v in vars(args).items() if "__" not in k})

            if args.experiment == "single_user":
                if args.blackbox > 0:
                    return self.run_bb_single_user(args, logger)
                else:
                    return self.run_wb_single_user(args, logger)

            elif args.experiment == "single_user_restrict":
                if args.blackbox > 0:
                    return self.run_bb_single_user_restricted_items(args, logger)
                else:
                    raise Exception("Invalid args - single_user_restrict is only for bb")

            elif args.experiment == "segment_restrict":
                if args.blackbox > 0:
                    return self.run_bb_segment_restricted_attack(args, logger)
                else:
                    raise Exception("Invalid args - segment_restrict is only for bb")

            elif args.experiment == "general_restrict":
                if args.blackbox > 0:
                    return self.run_bb_general_restricted_attack(args, logger)
                else:
                    raise Exception("Invalid args - general_restrict is only for bb")

            elif args.experiment == "restrict_attack_rate":
                return self.run_restrict_attack_rate(args, logger)

            elif args.experiment == "general":
                if args.blackbox > 0:
                    return self.run_bb_general_pop(args, logger)
                else:
                    return self.run_wb_general_pop(args, logger)

            elif args.experiment == "segment":
                if args.blackbox > 0:
                    return self.run_bb_segment_attack(args, logger)
                else:
                    return self.run_wb_segment_attack(args, logger)

            elif args.experiment == "baseline":
                return self.run_baseline(args, logger)

if __name__ == '__main__':
    parser = ArgumentParser(description="Experiments")

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--experiment', type=str, default="general_restrict")
    parser.add_argument('--blackbox', type=int, default=1)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--epsilon', type=int, default=1/255)
    parser.add_argument('--gamma', type=int, default=7)
    parser.add_argument('--user', type=int, default=80317)
    parser.add_argument('--item', type=int, default=190377)
    parser.add_argument('--from-rank', type=int, default=100000)
    parser.add_argument('--do-pca', type=int, default=0)
    parser.add_argument('--by-rank', type=int, default=1)
    parser.add_argument('--n-components', type=int, default=150)
    parser.add_argument('--rank-distribution', type=str, default='normal')
    parser.add_argument('--examples', type=int, default=32)

    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--k2', type=int, default=10)
    parser.add_argument('--algorithm', type=str, default='vbpr') # vbpr, vbprc, deepstyle
    parser.add_argument('--experiment_name', type=str, default='exp_defualt')

    args = parser.parse_args()

    dataset_name = "Clothing_Shoes_and_Jewelry"

    dataset = RecSysDataset(dataset_name)

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

    model.load(f'../data/dataset/{dataset_name}/models/{args.algorithm}_resnet50.pth')

    # args.user = model.add_fake_user_by_item(189844)

    print(args)

    exp = Experimentation(model, dataset_name)

    exp.run(args.experiment_name, args)

    
    

        