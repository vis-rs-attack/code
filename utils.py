import gzip, os, pickle, random, csv
from collections import defaultdict
import numpy as np

from PIL import Image
import torch.nn as nn
import torch

# from models import ImageModel

def in_jupyter():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
          return True
    except:
      pass
    return False
    
def import_tqdm():
  if in_jupyter():
    from tqdm import tqdm_notebook as tqdm
  else:
    from tqdm import tqdm
  return tqdm

top_category = "Clothing, Shoes & Jewelry"

image_folder = "../data/images/"

path = "../"

review_file = path + "data/reviews_Clothing_Shoes_and_Jewelry.json.gz"
metadata_file = path + "data/meta_Clothing_Shoes_and_Jewelry.json.gz"

architecture = "resnet50"

if architecture == "vgg16":
    # image_features_file = path + "data/vgg16-feats.gz"
    image_features_matrix = path + "data/items_vgg16_v1.pt"
    model_file = path + "data/vbpr_vgg16_v1.pth"
    image_feat_size = 4096

elif architecture == "resnet50":
    # image_features_file = path + "data/resnet50-feats.gz"
    image_features_matrix = path + "data/items_resnet50_v3.pt"
    model_file = path + "data/vbpr_resnet50_v3.pth"
    image_feat_size = 2048

else:
    print("Unknown architecture")
    exit(1)

items_csv = path + "data/items_c.csv"
users_items_csv = path + "data/users_items.csv"
testset_file = path + "data/testset.text"
validation_file = path + "data/validation.text"

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def load_itemset_with_image():
    itemset = set()
    f = open(image_features_file, 'rb')
    while True: 
        asin = f.read(10).decode('utf8')
        if asin == '': break
        f.read(4*image_feat_size)
        itemset.add(asin)
    return itemset

def load_itemset():
    itemset = set()
    with open(path + "data/items.csv") as f:
        f.readline() #skip the header
        reader = csv.reader(f)
        for row in reader:
            asin, _, _ = row
            itemset.add(asin)
    return itemset

def get_subcategory(categories):
    for cats in categories:
        if cats[0] == top_category:
            return cats[1]
    return None

def build_items_categories():
    items = []
    print("Reading itemset...")
    # itemset = load_itemset_with_image()
    itemset = load_itemset()
    missing_url, not_exist = 0, 0
    print("Reading metadata json...")
    with gzip.open(metadata_file) as fin:
        for line in fin:
            obj = eval(line)
            asin = obj['asin']
            if not 'imUrl' in obj:
                missing_url += 1
                continue
            if asin not in itemset:
                not_exist += 1
                continue
            items.append((asin, obj['categories']))

    print(f"Store {items_csv} file...")
    items = sorted(items)
    with open(items_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow("asin,categories".split(','))
        writer.writerows(items)

def build_items_data():
    items = []
    print("Reading itemset...")
    # itemset = load_itemset_with_image()
    itemset = load_itemset()
    missing_url, not_exist, not_jpg = 0, 0, 0
    print("Reading metadata json...")
    with gzip.open(metadata_file) as fin:
        for line in fin:
            obj = eval(line)
            asin = obj['asin']
            if not 'imUrl' in obj:
                missing_url += 1
                continue
            if asin not in itemset:
                not_exist += 1
                continue
            url = obj['imUrl']
            filename = url[url.rfind('/')+1:]
            if not filename.endswith(".jpg"):
                not_jpg += 1
                continue
            cat = get_subcategory(obj['categories'])
            items.append((asin, cat, filename))

    print(f"Store {items_csv} file...")
    items = sorted(items)
    with open(items_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow("asin,sub-category,image-filename".split(','))
        writer.writerows(items)

    print({
        "valid_items": len(items), 
        "missing_url": missing_url, 
        "not_exist": not_exist,
        "not_jpg": not_jpg
    })

def read_item_data():
    items_data, items_idx = [], {}
    with open(items_csv) as f:
        f.readline() #skip the header
        reader = csv.reader(f)
        for row in reader:
            asin, category, filename = row
            items_data.append((asin, category, filename))
            items_idx[asin] = len(items_idx)
    return items_data, items_idx

def build_users_items_data():

    _, items_idx = read_item_data()

    errors = 0
    user_items = defaultdict(list)
    print("Reading review json...")
    with gzip.open(review_file) as fin:
      for line in fin:
          try:
              obj = eval(line)
              if obj['overall'] > 3:
                  asin = obj['asin']
                  if asin in items_idx: # skip items on in our itemset
                      user = obj['reviewerID']
                      user_items[user].append(asin)
          except:
              errors += 1

    print(f"Total users: {len(user_items)}")
    user_items = { user:user_items[user] for user in user_items if len(user_items[user]) >= 5 }
    print(f"Users with 5 or more items: {len(user_items)}")

    print(f"Store {users_items_csv} file...")
    users = sorted(user_items.keys())
    with open(users_items_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow("user,list-of-items".split(','))
        for user in users:
            writer.writerow([user] + user_items[user])

def read_user_item_data():
    items_data, items_idx = read_item_data()

    user_items = []
    with open(users_items_csv) as f:
        f.readline() #skip the header
        reader = csv.reader(f)
        for row in reader:
            _, items = row[0], map(lambda i: items_idx[i], row[1:])
            user_items.append(tuple(items))
    
    item_category = []
    category_idx = {}
    for _, category, _ in items_data:
        category_id = category_idx.setdefault(category, len(category_idx))
        item_category.append(category_id)

    return user_items, item_category

def load_image_features():

    print("Reading image features...")

    _, items_idx = read_item_data()

    image_features = {}
    with gzip.open(image_features_file, 'rb') as f:
        while True: 
            asin = f.read(10).decode('utf8')
            if asin == '':
                break
            features_bytes = f.read(4*image_feat_size)
            if asin in items_idx:
                features = np.frombuffer(features_bytes, dtype=np.float32)
                image_features[items_idx[asin]] = features

    return image_features

def tmp(fni, fno):
    print("Reading image features...", end='', flush=True)
    M = torch.load(fni)
    torch.save(60*M, fno)

def store_image_features_matrix():

    print("Reading image features...")

    _, items_idx = read_item_data()

    image_features = torch.zeros(len(items_idx), image_feat_size, dtype=torch.float32)
    # image_features = np.zeros((len(items_idx), image_feat_size), dtype=np.float32)
    cnt = 0
    with gzip.open(image_features_file, 'rb') as f:
        while True: 
            asin = f.read(10).decode('utf8')
            if asin == '':
                break
            features_bytes = f.read(4*image_feat_size)
            if asin in items_idx:
                features = np.frombuffer(features_bytes, dtype=np.float32)
                
                image_features[items_idx[asin]] = torch.tensor(features, dtype=torch.float32)
                cnt += 1
        print(f"total items read {cnt}")
    torch.save(image_features, image_features_matrix)

    return image_features

def train_test_split(user_items):
    return [random.choice(items) for items in user_items]

def train_test_validation_split():
    print("Read data and split...")
    user_items = read_user_item_data()

    testset = train_test_split(user_items)
    validation = train_test_split(user_items)

    with open(testset_file, "w") as f:
        for i in testset:
            f.write(f"{i}\n")
    with open(validation_file, "w") as f:
        for i in validation:
            f.write(f"{i}\n")

class InvalidImage(Exception):
    pass

def create_image_features():

    image_folder = "../data/Clothing_Shoes_and_Jewelry"

    with open("data.pkl", "rb") as f:
        items_image, _ = pickle.load(f)

    image_model = ImageModel('vgg16')

    with open('new_image_features.b', "wb") as f:
        batch, items = [], []
        for item, name in tqdm(items_image.items()):
            try:
                filename = os.path.join(image_folder, name)
                img = Image.open(filename)
                try:
                    img_t = image_model.transform(img)
                except:
                    raise InvalidImage()
                
                if img_t.shape != (3, 224, 224):
                    raise InvalidImage()
                batch.append(img_t)
                if len(batch) == 128:
                    batch_t = torch.stack(batch)
                    features = image_model.get_features(batch_t)
                    for feat, item in zip(batch, items):
                        f.write(item.encode('utf8'))
                        f.write(feat.numpy().tobytes())
                    batch, items = [], []
            except InvalidImage:
                print(f"InvalidImage error processing {name}")
    

if __name__ == "__main__":
    # build_items_data()
    # build_items_categories()
    # build_users_items_data()
    # load_image_features()
    # load_image_features_matrix()
    # train_test_validation_split()
    # store_image_features_matrix()

    read_user_item_data()

    # run_preperp()
    # index_data()
    # train_test_validation_split()
    # load_image_features()

    # tmp("../data/items_vgg16_v1.pt", "../data/items_vgg16_v2.pt")
    




    