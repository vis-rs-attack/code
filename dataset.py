import random, os, csv

import torch
from torch.utils.data import Dataset

def train_test_split(user_items):
    return [random.choice(items) for items in user_items]

def train_test_validation_split(dataset_name):
    dataset_path = f"../data/dataset/{dataset_name}"
    
    user_items, _ = read_user_item_data(dataset_path)

    testset = train_test_split(user_items)
    validation = train_test_split(user_items)

    testset_file = os.path.join(dataset_path, "testset.txt")
    with open(testset_file, "w") as f:
        for i in testset:
            f.write(f"{i}\n")

    validation_file = os.path.join(dataset_path, "validation.txt")
    with open(validation_file, "w") as f:
        for i in validation:
            f.write(f"{i}\n")

def read_item_data(dataset_path):
    items_data, items_idx = [], {}
    with open(os.path.join(dataset_path, "items_filtered.csv")) as f:
        f.readline() #skip the header
        reader = csv.reader(f)
        for row in reader:
            asin, category, filename = row
            items_data.append((asin, category, filename))
            items_idx[asin] = len(items_idx)
    return items_data, items_idx

def read_user_item_data(dataset_path):
    items_data, items_idx = read_item_data(dataset_path)

    user_items = []
    with open(os.path.join(dataset_path, "users_items.csv")) as f:
        f.readline() #skip the header
        reader = csv.reader(f)
        for row in reader:
            items = row[1:]
            items = tuple(items_idx[i] for i in items if i in items_idx)
            user_items.append(items)
    
    item_category = []
    category_idx = {}
    for _, category, _ in items_data:
        category_id = category_idx.setdefault(category, len(category_idx))
        item_category.append(category_id)

    return user_items, item_category

def read_list_from_file(filename):
    lst = []
    with open(filename) as f:
        for num in f:
            lst.append(int(num.strip()))
    return lst

def read_testset(testset_file):
    return read_list_from_file(testset_file)

def read_validation(validation_file):
    return read_list_from_file(validation_file)

def load_image_features_matrix(dataset_path):
    print("Reading image features...", end='', flush=True)
    M = torch.load(os.path.join(dataset_path, "models", "items_resnet50_v3.pt"))
    print("done!")
    if torch.cuda.is_available():
        M = M.cuda()
    return M

class Corpus():
    def __init__(self, dataset_path):

        self.user_items, self.item_category = read_user_item_data(dataset_path)
        self.testset = read_testset(os.path.join(dataset_path, "testset.txt"))
        self.validation = read_validation(os.path.join(dataset_path, "validation.txt"))
        self.image_features = load_image_features_matrix(dataset_path)

    def is_in_evaluation_sets(self, user, item):
        return self.testset[user] == item or self.validation[user] == item

    def is_in_user_list(self, user, item):
        return item in self.user_items[user]

    def get_image_features(self, vi):
        return self.image_features[vi]

    def get_item_category(self, i):
        return self.item_category[i]

class RecSysDataset(Dataset):
    def __init__(self, dataset_name):
        dataset_path = f"../data/dataset/{dataset_name}"
        self.corpus = Corpus(dataset_path)
        self.data = []
        for u, items in enumerate(self.corpus.user_items):
            for i in items:
                if self.corpus.is_in_evaluation_sets(u, i):
                    continue
                self.data.append((u, i))
        self.size = len(self.data)

        self.n_users = len(self.corpus.user_items)
        self.n_items = len(self.corpus.image_features)
        self.n_categories = max(self.corpus.item_category)+1

    def __getitem__(self, idx):
        u, i = self.data[idx]
        while True:
            j = random.randint(0, self.n_items-1)
            if self.corpus.is_in_user_list(u, j):
                continue
            break
        return u, i, j

    def __len__(self):
        return self.size

    def get_validation(self, size=-1):
        return RecSysTestset(self.corpus, self.corpus.validation, size)

    def get_testset(self, size=-1):
        return RecSysTestset(self.corpus, self.corpus.testset, size)

class RecSysTestset(Dataset):
    def __init__(self, corpus, data, size=-1):
        self.corpus = corpus
        self.data = data
        if size < 0:
            size = len(self.data)
        elif size < len(self.data):
            self.data = self.data[:size]
        else:
            print("size can't be larger than test set.")
            print("setting size to the test set size")
            size = len(self.data)
        self.size = size

        self.n_users = len(self.corpus.user_items)
        self.n_items = len(self.corpus.image_features)

    def __getitem__(self, u):
        i = self.data[u]
        while True:
            j = random.randint(0, self.n_items-1)
            if self.corpus.is_in_user_list(u, j):
                continue
            return u, i, j

    def __len__(self):
        return self.size

if __name__ == '__main__':
    dataset_name = "Clothing_Shoes_and_Jewelry"
    dataset_path = f"../data/dataset/{dataset_name}"
    train_test_validation_split(dataset_name)
