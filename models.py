import numpy as np
import torch.nn as nn
import torch

import utils

def l2(*tensors):
    return sum([tensor.pow(2).sum() for tensor in tensors])/2

def inner(a, b):
    return (a * b).sum(dim=1)


class Recommender(nn.Module):
    def create_param(self, *size):
        w = nn.Parameter(torch.rand(*size))
        if len(size) > 1:
            nn.init.xavier_uniform_(w)
        else:
            nn.init.zeros_(w)
        return w

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=utils.get_device()))
        if torch.cuda.is_available():
            self = self.cuda()
        self.eval()

    def score(self, u):
        i = torch.LongTensor(range(self.n_items))
        return -self.pointwise_forward(u, i, self.F)

    def score_items_user(self, u):
        i = torch.LongTensor(range(self.n_items))
        return -self.pointwise_forward(u, i, self.F)

    def score_users_item(self, users, i, fi=None):
        return -self.pointwise_forward(users, [i], fi)

    def score_user_item(self, u, i, fi=None):
        return -self.pointwise_forward([u], [i], fi)


class VBPR(Recommender):
    def __init__(self, n_users, n_items, F, k=10, k2=10, lambda_w=0.01, lambda_b=0.01, lambda_e=0.0):
        super(VBPR, self).__init__()

        self.n_users = n_users
        self.n_items = n_items

        self.F = F/60
        self.k = k
        self.k2 = k2

        self.lambda_w = lambda_w
        self.lambda_b = lambda_b
        self.lambda_e = lambda_e

        self.Bi = self.create_param(n_items)
        self.Gu = self.create_param(n_users, k)
        self.Gi = self.create_param(n_items, k)
        self.Tu = self.create_param(n_users, k2)
        self.E = self.create_param(F.shape[1], k2)
        self.Bp = self.create_param(F.shape[1], 1)

    def add_fake_user_by_item(self, i):
        ti = (self.F[[i]] @ self.E).detach()
        gi = self.Gi[[i]].detach()
        self.Tu = nn.Parameter(torch.cat((self.Tu.detach(), ti)))
        self.Gu = nn.Parameter(torch.cat((self.Gu.detach(), gi)))

        self.n_users += 1

        return self.n_users-1

    
    def score_similar_items(self, i):
        Ti = (self.F @ self.E).detach()
        ti = Ti[[i]]
        gi = self.Gi[[i]].detach()
        scores = Ti @ ti.T + self.Gi @ gi.T
        return scores.detach().squeeze()


    def add_fake_user_by_items(self, items):
        ti = (self.F[items] @ self.E).detach().mean(axis=0, keepdim=True)
        gi = self.Gi[items].detach().mean(axis=0, keepdim=True)
        self.Tu = nn.Parameter(torch.cat((self.Tu.detach(), ti)))
        self.Gu = nn.Parameter(torch.cat((self.Gu.detach(), gi)))

        self.n_users += 1

        return self.n_users-1


    def add_fake_user_by_other_and_add_items(self, u, user_items, items):
        tu = self.Tu[[u]].detach()
        gu = self.Gu[[u]].detach()

        ti = (self.F[items] @ self.E).detach().sum(axis=0)
        gi = self.Gi[items].detach().sum(axis=0)

        n = len(user_items)
        m = len(items)

        tu = (n*tu+ti)/(n+m)
        gu = (n*gu+gi)/(n+m)

        self.Tu = nn.Parameter(torch.cat((self.Tu.detach(), tu)))
        self.Gu = nn.Parameter(torch.cat((self.Gu.detach(), gu)))

        self.n_users += 1

        return self.n_users-1


    def score_users(self, u):
        gamma_u = self.Gu[u]
        theta_u = self.Tu[u]

        beta_i = self.Bi
        gamma_i = self.Gi
        feat_i = self.F

        Xui = beta_i + \
            gamma_u @ gamma_i.T + \
            theta_u @ feat_i.mm(self.E).T + \
            feat_i.mm(self.Bp).squeeze()
        
        return Xui


    def pointwise_forward(self, u, i, fi=None):
        gamma_u = self.Gu[u]
        theta_u = self.Tu[u]

        beta_i = self.Bi[i]
        gamma_i = self.Gi[i]
        feat_i = self.F[i] if fi is None else fi

        Xui = beta_i + \
            inner(gamma_u, gamma_i) + \
            inner(theta_u, feat_i.mm(self.E)) + \
            feat_i.mm(self.Bp).squeeze()
        
        return -Xui

    def forward(self, u, i, j, fi=None, fj=None):
        gamma_u = self.Gu[u]
        theta_u = self.Tu[u]

        beta_i = self.Bi[i]
        gamma_i = self.Gi[i]
        feat_i = self.F[i] if fi is None else fi

        beta_j = self.Bi[j]
        gamma_j = self.Gi[j]
        feat_j = self.F[j] if fj is None else fj

        gamma_diff = gamma_i - gamma_j
        feat_diff = feat_i - feat_j

        Xuij = beta_i - beta_j + \
            inner(gamma_u, gamma_diff) + \
            inner(theta_u, feat_diff.mm(self.E)) + \
            feat_diff.mm(self.Bp).squeeze()

        log_likelihood = nn.functional.logsigmoid(Xuij).sum()

        reg = l2(gamma_u, gamma_i, gamma_j, theta_u) * self.lambda_w + \
            l2(beta_i, beta_j) * self.lambda_b + \
            l2(self.E, self.Bp) * self.lambda_e

        loss = -log_likelihood + reg
        auc = (Xuij > 0).float().sum()
        
        return loss, auc


class VBPRC(Recommender):
    def __init__(
            self, n_users, n_items, n_categories, F, IC, 
            k, k2, lambda_w=0.01, lambda_b=0.01, lambda_e=0.0):

        super(VBPRC, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories

        self.F = F/60
        self.IC = torch.LongTensor(IC)
        self.k = k
        self.k2 = k2

        self.lambda_w = lambda_w
        self.lambda_b = lambda_b
        self.lambda_e = lambda_e

        self.Bi = self.create_param(n_items)
        self.Gu = self.create_param(n_users, k)
        self.Gi = self.create_param(n_items, k)
        self.Tu = self.create_param(n_users, k2)
        self.Ic = self.create_param(n_categories, k2)
        self.E = self.create_param(F.shape[1], k2)
        self.Bp = self.create_param(F.shape[1], 1)

    def pointwise_forward(self, u, i, fi=None):
        gamma_u = self.Gu[u]
        theta_u = self.Tu[u]

        ci = self.IC[i]

        beta_i = self.Bi[i]
        gamma_i = self.Gi[i]
        cf_i = self.Ic[ci]
        feat_i = self.F[i] if fi is None else fi

        Xui = beta_i + \
            inner(gamma_u, gamma_i) + \
            inner(theta_u, feat_i.mm(self.E) - cf_i) + \
            feat_i.mm(self.Bp).squeeze()
        
        return -Xui

    def forward(self, u, i, j, fi=None, fj=None):
        gamma_u = self.Gu[u]
        theta_u = self.Tu[u]

        ci = self.IC[i]
        cj = self.IC[j]

        beta_i = self.Bi[i]
        gamma_i = self.Gi[i]
        cf_i = self.Ic[ci]
        feat_i = self.F[i] if fi is None else fi

        beta_j = self.Bi[j]
        gamma_j = self.Gi[j]
        cf_j = self.Ic[cj]
        feat_j = self.F[j] if fj is None else fj

        gamma_diff = gamma_i - gamma_j
        feat_diff = feat_i - feat_j

        cf_diff = cf_i - cf_j

        Xuij = beta_i - beta_j + \
            inner(gamma_u, gamma_diff) + \
            inner(theta_u, feat_diff.mm(self.E) - cf_diff) + \
            feat_diff.mm(self.Bp).squeeze()

        log_likelihood = nn.functional.logsigmoid(Xuij).sum()

        reg = l2(gamma_u, gamma_i, gamma_j, theta_u) * self.lambda_w + \
            l2(beta_i, beta_j) * self.lambda_b + \
            l2(self.E, self.Bp, cf_i, cf_j) * self.lambda_e

        loss = -log_likelihood + reg
        auc = (Xuij > 0).float().sum()
        
        return loss, auc


class DeepStyle(Recommender):
    def __init__(
            self, n_users, n_items, n_categories, F, IC, k=10, lambda_w=0.01, lambda_e=0.01):

        super(DeepStyle, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories

        self.F = F/60
        self.IC = torch.LongTensor(IC)
        self.k = k

        self.lambda_w = lambda_w
        self.lambda_e = lambda_e

        self.Pu = self.create_param(n_users, k)
        self.Qi = self.create_param(n_items, k)
        self.Bi = self.create_param(n_items)
        self.E = self.create_param(F.shape[1], k)
        self.Bp = self.create_param(F.shape[1], 1)
        self.Ic = self.create_param(n_categories, k)
        self.Bc = self.create_param(n_categories)

    def pointwise_forward(self, u, i, vi=None):
        ci = self.IC[i]

        if vi is None: vi = self.F[i]

        pu = self.Pu[u]
        qi = self.Qi[i]
        bi = self.Bi[i]
        ii = self.Ic[ci]
        bic = self.Bc[ci]
        
        Yui = bi + bic + inner(pu, vi.mm(self.E)+qi-ii) + vi.mm(self.Bp).squeeze()

        return -Yui

    def forward(self, u, i, j, vi=None, vj=None):
        if vi is None: vi = self.F[i]
        if vj is None: vj = self.F[j]

        ci = self.IC[i]
        cj = self.IC[j]

        pu = self.Pu[u]

        qi, qj = self.Qi[i], self.Qi[j]
        bi, bj = self.Bi[i], self.Bi[j]
        ii, ij = self.Ic[ci], self.Ic[cj]
        bic, bjc = self.Bc[ci], self.Bc[cj]

        dv, dq, di = (vi - vj), (qi - qj), (ii - ij)
        Yuij = (bi - bj) + (bic - bjc) + inner(pu, dv.mm(self.E)+dq-di) + dv.mm(self.Bp).squeeze()

        log_likelihood_eq = -(1+torch.exp(-Yuij)).log().sum()
        # log_likelihood = nn.functional.logsigmoid(Yuij).sum()

        reg = l2(pu, qi, qj, ii, ij) * self.lambda_w + \
            l2(bi, bj, bic, bjc) * self.lambda_w + \
            l2(self.E, self.Bp) * self.lambda_e

        loss = -log_likelihood_eq + reg
        auc = (Yuij > 0).float().sum()
        
        return loss, auc


class BPR(Recommender):
    def __init__(self, n_users, n_items, k=10, lambda_w=0.01, lambda_b=0.01):
        super(BPR, self).__init__()

        self.n_users = n_users
        self.n_items = n_items

        self.k = k

        self.lambda_w = lambda_w
        self.lambda_b = lambda_b

        self.Bi = self.create_param(n_items)
        self.Gu = self.create_param(n_users, k)
        self.Gi = self.create_param(n_items, k)

    def pointwise_forward(self, u, i):
        gamma_u = self.Gu[u]

        beta_i = self.Bi[i]
        gamma_i = self.Gi[i]

        Xui = beta_i + inner(gamma_u, gamma_i)
        
        return -Xui

    def forward(self, u, i, j):
        gamma_u = self.Gu[u]

        beta_i = self.Bi[i]
        gamma_i = self.Gi[i]

        beta_j = self.Bi[j]
        gamma_j = self.Gi[j]

        gamma_diff = gamma_i - gamma_j

        Xuij = beta_i - beta_j + inner(gamma_u, gamma_diff)

        log_likelihood = nn.functional.logsigmoid(Xuij).sum()

        reg = l2(gamma_u, gamma_i, gamma_j) * self.lambda_w + \
            l2(beta_i, beta_j) * self.lambda_b

        loss = -log_likelihood + reg
        auc = (Xuij > 0).float().sum()
        
        return loss, auc
        

from torchvision import transforms, models

class ImageModel:
    def __init__(self, architecture, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transforms = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        if architecture == "vgg16":
            model = models.vgg16(pretrained=True)
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
            self.image_feat_size = 4096

        elif architecture == "resnet50":
            self.image_feat_size = 2048
            class Squeeze(nn.Module):
                def forward(self, x):
                    return x.view(-1, 2048)
            model = models.resnet50(pretrained=True)
            modules = list(model.children())[:-1] + [Squeeze()]
            model = nn.Sequential(*modules)

        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        self.model = model
        self.architecture = architecture

    def get_features(self, imgs):
        return self.model(imgs)

    def transform(self, img):
        return self.transforms(img)

    @property
    def Dim(self):
        return 4096 if self.architecture == "vgg16" else 2048
