import gzip, os, pickle, random
from collections import defaultdict
import numpy as np

from PIL import Image
import torch.nn as nn
import torch

from tqdm import tqdm
import concurrent.futures

import utils
from models import ImageModel

path = "../"

image_folder = path + "data/Clothing_Shoes_and_Jewelry"

input_size = (3, 244, 244)

use_gpu = torch.cuda.is_available()

items_data, items_idx = utils.read_item_data()
user_items = utils.read_user_item_data()

im = ImageModel('resnet50')

image_feat_size = im.image_feat_size

class InvalidImage(Exception):
    pass

def work(msg):
    item, _, name = msg
    try:
        filename = os.path.join(image_folder, name)
        img = Image.open(filename)
        img_t = im.transform(img)
        return item, img_t
    except InvalidImage:
        print(f"InvalidImage error processing {name}")

def extract(chunk):
    with concurrent.futures.ThreadPoolExecutor() as executor:
      res = list(executor.map(work, chunk))
    items, batch = zip(*res)

    batch = torch.stack(batch)
    if use_gpu:
      batch = batch.cuda()
    features = im.get_features(batch)

    return items, features.squeeze()

chunksize=256
chunks = [items_data[i*chunksize:(i+1)*chunksize] for i in range(1+len(items_data)//chunksize)]

image_features = torch.zeros(len(items_idx), image_feat_size, dtype=torch.float32)

cnt = 0
for chunk in tqdm(chunks):
    items, features = extract(chunk)
    for item, feat in zip(items, features):
        feat = feat.view(-1)
        image_features[items_idx[item]] = feat
        cnt += 1
print(cnt)

torch.save(image_features, "items_resnet50_v2.pt")