import multiprocessing as mp

import gzip, os, pickle, random
from collections import defaultdict
import numpy as np

from PIL import Image

import torch.nn as nn
import torch, sys

from tqdm import tqdm

import utils

from models import ImageModel

image_folder = "../data/Clothing_Shoes_and_Jewelry"

class InvalidImage(Exception):
    pass

items_data, items_idx = utils.read_item_data()

images = [(a[0], a[2]) for a in items_data]

image_model = ImageModel("vgg16")

def worker(msg, q):
    try:
        item, name = msg
        filename = os.path.join(image_folder, name)
        img = Image.open(filename)
        try:
            img_t = image_model.transform(img)
        except:
            raise InvalidImage()
        if img_t.shape != (3, 224, 224):
            raise InvalidImage()
        batch_t = torch.unsqueeze(img_t, 0)
        features = image_model.get_features(batch_t)
        q.put((item, features))
        return (item, features)
    except InvalidImage:
        print(f"InvalidImage error processing {name}")
        return "clom"

def listener(q):
    '''listens for messages on the q, writes to file. '''
    with open('../data/new_image_features_8.b', "wb") as f:
        while 1:
            try:
                msg = q.get()
                if msg is None:
                    print("Done!")
                    break
                item, feat = msg
                f.write(item.encode('utf8'))
                f.write(feat.numpy().tobytes())
            except Exception as e:
                print(e)

def load_exists():
    existing_items = set()
    for i in [2,3,4,5,6,7]:
        try:
            filename = f"../data/new_image_features_{i}.b"
            print(f"Reading {filename} ...")
            f = open(filename, 'rb')
            while True: 
                asin = f.read(10).decode('utf8')
                if asin == '': break
                f.read(4*image_model.Dim)
                existing_items.add(asin)
        except FileNotFoundError as e:
            print(e)
    return existing_items

def main():
    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(6)
    existing_items = load_exists()

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    for item, name in tqdm(images):
        if item in existing_items:
            continue
        job = pool.apply_async(worker, ((item, name), q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in tqdm(jobs): 
        job.get()

    #now we are done, kill the listener
    q.put(None)
    pool.close()
    pool.join()

if __name__ == "__main__":
   main()