import random, os, math

from pprint import pprint

from experiments import Experimentation
from dataset import RecSysDataset
from models import VBPR, DeepStyle

import numpy as np
import torch

class Args:
    def __init__(self, args):
        self.__dict__.update(args)

def load_model(dataset_name, algo_name, dataset):
  if algo_name == "vbpr":
    model = VBPR(dataset.n_users, dataset.n_items, dataset.corpus.image_features)
  elif algo_name == "deepstyle":
    model = DeepStyle(
      dataset.n_users, dataset.n_items, dataset.n_categories, 
      dataset.corpus.image_features, dataset.corpus.item_category
    )

  model.load(f'../data/dataset/{dataset_name}/models/{algo_name}_resnet50.pth')

  return model

def run_grid_search():
  from exps import gen_conf, grid_config_v1, base_config_v1, alreay_perfromed

  dataset_name = 'Electronics'
  algo_name = 'vbpr'

  experiments = Experimentation(load_model(dataset_name, algo_name), dataset_name)

  for name, args in gen_conf(grid_config_v1):
    if name not in alreay_perfromed:
      print(name)
      args.update(base_config_v1)
      pprint(args)
      experiments.run(name, Args(args))

  # with open("exp/exps.jsonl", "r") as f:
  #   for line in f:
  #     line = line.strip()
  #     if line.startswith("#"):
  #       continue
  #     args = Args(eval(line))
  #     exp.run(args)

default_args = {
  'k': 10,
  'k2': 10,
  'algorithm': 'vbpr',
  'do_pca': 0,
  'n_components': 128,
  'rank_distribution': 'uniform',
}

def run_from_rank_exp():

  dataset_name = 'Clothing_Shoes_and_Jewelry'
  algo_name = 'vbpr'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  random.seed(seed)

  args = {
    'seed': seed,
    'experiment': 'single_user',
    'dataset_name': dataset_name,
    'steps': 20,
    'epsilon': 1/255,
    'gamma': 7,
    'by_rank': 1,
    'examples': 32,
  }

  args.update(default_args)

  exp_folder = f"exp_from_rank_range"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  random_users = random.sample(range(exp.model.n_users), k=50)
  rank_ranges = [1, 1000, 10000, 100000, 455412]
  for min_rank, max_rank in zip(rank_ranges[:-1], rank_ranges[1:]):
    from_rank = random.randint(min_rank, max_rank)
    args['from_rank'] = from_rank
    for blackbox in [0, 1]:
      args['blackbox'] = blackbox
      for user in random_users:
        args['user'] = user

        name = f"from_rank[{from_rank}, {blackbox}, {user}]"
        exp.run(f"{exp_folder}/{name}", Args(args))
        # im.save(f"exp/{name}.jpeg", "JPEG")

def run_main_table_exp():

  dataset_name = 'Electronics'
  algo_name = 'deepstyle'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  random.seed(seed)

  args = {
    'seed': seed,
    'experiment': 'single_user',
    'dataset_name': dataset_name,
    'steps': 20,
    'epsilon': 1/255,
    'gamma': 7,
    'examples': 32,
  }

  args.update(default_args)

  exp_folder = f"exp_{dataset_name}_{algo_name}"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  repeat = 30

  random_users = random.sample(range(exp.model.n_users), k=repeat)
  random_items = random.sample(range(exp.model.n_items), k=repeat)
  for u, i in zip(random_users, random_items):
    args['user'] = u
    args['item'] = i
    for blackbox in [0, 1]:
      args['blackbox'] = blackbox
      if blackbox:
        for by_rank in [0, 1]:
          args['by_rank'] = by_rank
          name = f"main[{blackbox}:{by_rank}, {u}, {i}]"
          im = exp.run(f"{exp_folder}/{name}", Args(args))
          im.save(f"{exp_folder}/{name}.jpeg", "JPEG")
      else:
        name = f"main[{blackbox}, {u}, {i}]"
        im = exp.run(f"{exp_folder}/{name}", Args(args))
        im.save(f"{exp_folder}/{name}.jpeg", "JPEG")


def run_bb_examples_per_step_exp():

  dataset_name = 'Clothing_Shoes_and_Jewelry'
  algo_name = 'vbpr'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  random.seed(seed)

  args = {
    'seed': seed,
    'experiment': 'single_user',
    'dataset_name': dataset_name,
    'epsilon': 1/255,
    'gamma': 7,
    'blackbox': 1,
    'by_rank': 1,
    'steps': 20,
  }

  args.update(default_args)

  exp_folder = f"exp_ex2step"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  repeat = 100
  budget = 4096

  # random_users = random.sample(range(exp.model.n_users), k=repeat)
  # random_items = random.sample(range(exp.model.n_items), k=repeat)
  # for u, i in zip(random_users, random_items):
  #   args['user'] = u
  #   args['item'] = i
  #   for p in range(1, int(math.log2(budget))):
  #     steps = 2**p
  #     examples = budget // steps
  #     args['examples'] = examples
  #     args['steps'] = steps
  #     name = f"main[{examples}, {steps}, {u}, {i}]"
  #     im = exp.run(f"{exp_folder}/{name}", Args(args))
  #     im.save(f"{exp_folder}/{name}.jpeg", "JPEG")

  random_users = random.sample(range(exp.model.n_users), k=repeat)
  random_items = random.sample(range(exp.model.n_items), k=repeat)
  for u, i in zip(random_users, random_items):
    args['user'] = u
    args['item'] = i
    for examples in [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]:
      steps = args['steps']
      args['examples'] = examples
      name = f"main[{examples}, {steps}, {u}, {i}]"
      im = exp.run(f"{exp_folder}/{name}", Args(args))
      # im.save(f"{exp_folder}/{name}.jpeg", "JPEG")


def run_bb_single_user_restrict_exp():

  dataset_name = 'Clothing_Shoes_and_Jewelry'
  algo_name = 'vbpr'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  random.seed(seed)

  args = {
    'seed': seed,
    'experiment': 'single_user_restrict',
    'dataset_name': dataset_name,
    'epsilon': 1/255,
    'gamma': 7,
    'blackbox': 1,
    'by_rank': 1,
    'steps': 20,
    'examples': 32,
  }

  args.update(default_args)

  exp_folder = f"exp_restrict"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  user_items = """110680, 358983, 198980
50494, 181343, 78779
99346, 103652, 178736
116686, 105072, 175653
55125, 193068, 182942
5306, 316952, 79752
33936, 68717, 218461
126545, 446829, 40044
67013, 282366, 129428
63691, 427098, 177546
53075, 234852, 323772
120354, 230606, 408602
102734, 142737, 62278
108770, 442049, 284216
39755, 82429, 311327
126851, 199249, 121946
62468, 211997, 29955
46930, 17209, 65192
76465, 239979, 218461
116871, 266575, 41210
119028, 262485, 125305
28631, 416912, 112862
66150, 71677, 443957
18254, 206620, 247487
36941, 169961, 408602
18316, 249120, 294815
99064, 176886, 115981
12429, 26161, 132444
81050, 290802, 319654
104779, 67934, 171678
32834, 217176, 323772
119242, 165728, 129428
69804, 21000, 84504
92428, 57480, 48070
106196, 92620, 284216
78892, 353985, 443957
118248, 271611, 31783
19262, 369153, 276431
40651, 448284, 354434
12945, 403923, 319030
95660, 26658, 364632
9665, 337449, 2359
117812, 267384, 408602
111473, 379750, 71288
89651, 407907, 302410
43279, 43924, 302410
61884, 55078, 198980
73375, 452746, 1270
13199, 199837, 364632
46372, 201706, 98940
56907, 72103, 64782
41444, 418994, 354066
80070, 48648, 284216
83941, 361106, 282048
119670, 30279, 94791
26801, 167055, 291390
126695, 421024, 253221
72420, 312971, 101043
62522, 136923, 424684
58024, 8549, 132548
113417, 87558, 302410
68334, 50863, 190043
34143, 20684, 263813
8163, 169657, 136482
105516, 342821, 198980
120385, 251989, 155477
71919, 106883, 263813
120061, 290530, 431949
1840, 45189, 182933
12225, 78503, 307606
94333, 410342, 302410
110112, 279824, 111106
52274, 154682, 129428
93094, 173811, 358612
108114, 436320, 189303
102897, 443663, 182933
87576, 152149, 130704
81954, 354735, 129428
149, 60954, 52300
80202, 256969, 302410
64694, 304643, 52300
108535, 256138, 253697
113718, 257299, 129428
43664, 249473, 146615
31969, 174893, 104513
95719, 158561, 361643
42625, 56205, 107749
92227, 107229, 194319
114094, 43563, 447975
8255, 450688, 333316
25043, 227106, 96823
120195, 196420, 360
74384, 304817, 245385
29059, 356727, 245385
31275, 80284, 167168
105296, 89162, 243781
126817, 94328, 183185
18677, 183543, 198980
105274, 18042, 358612
71170, 257054, 44373"""

  for u_i_tag in user_items.split('\n'):
    u, i, i_tag = eval(u_i_tag)
    args['user'], args['item'], args['i_tag'] = u, i, i_tag
    name = f"main[{u}, {i}, {i_tag}]"
    im = exp.run(f"{exp_folder}/{name}", Args(args))

def run_bb_segment_restrict_exp():
  dataset_name = 'Clothing_Shoes_and_Jewelry'
  algo_name = 'vbpr'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  args = {
    'seed': seed,
    'experiment': 'segment_restrict',
    'dataset_name': dataset_name,
    'epsilon': 1/255,
    'gamma': 7,
    'by_rank': 1,
    "examples": 32,
    "steps": 20,
  }

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  args.update(default_args)

  exp_folder = f"exp_segment_rest_3_{dataset_name}_{algo_name}"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  random_items = random.sample(range(exp.model.n_items), k=60)
  for i in random_items:
    args['item'] = i
    for blackbox in [1]:
      args['blackbox'] = blackbox
      name = f"[{i, blackbox}]"
      exp.run(f"{exp_folder}/{name}", Args(args))
      exp.model = load_model(dataset_name, algo_name, dataset)

def run_segment_exp():

  dataset_name = 'Clothing_Shoes_and_Jewelry'
  algo_name = 'vbpr'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  random.seed(seed)

  args = {
    'seed': seed,
    'experiment': 'segment',
    'dataset_name': dataset_name,
    'epsilon': 1/255,
    'gamma': 7,
    'by_rank': 1,
    "examples": 32,
    "steps": 20,
  }

  args.update(default_args)

  exp_folder = f"exp_segment_3_{dataset_name}_{algo_name}"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  random_items = random.sample(range(exp.model.n_items), k=60)
  for i in random_items:
    args['item'] = i
    for blackbox in [0,1]:
      args['blackbox'] = blackbox
      name = f"[{i, blackbox}]"
      exp.run(f"{exp_folder}/{name}", Args(args))
      exp.model = load_model(dataset_name, algo_name, dataset)


def run_general_pop_exp():

  dataset_name = 'Clothing_Shoes_and_Jewelry'
  algo_name = 'vbpr'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  args = {
    'seed': None,
    'experiment': 'general',
    'dataset_name': dataset_name,
    'epsilon': 1/255,
    'gamma': 7,
    'by_rank': 1,
    "examples": 32,
    "steps": 30,
  }

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  args.update(default_args)

  exp_folder = f"exp_genpop_pre25_{dataset_name}_{algo_name}"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  random_items = random.sample(range(exp.model.n_items), k=30)
  for i in random_items:
    args['item'] = i
    for blackbox in [0, 1]:
      args['blackbox'] = blackbox
      name = f"[{i, blackbox}]"
      exp.run(f"{exp_folder}/{name}", Args(args))
      exp.model = load_model(dataset_name, algo_name, dataset)

def run_general_pop_restrict_exp():
  dataset_name = 'Clothing_Shoes_and_Jewelry'
  algo_name = 'vbpr'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  args = {
    'seed': None,
    'experiment': 'general_restrict',
    'dataset_name': dataset_name,
    'epsilon': 1/255,
    'gamma': 7,
    'by_rank': 1,
    "examples": 32,
    "steps": 20,
  }

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  args.update(default_args)

  exp_folder = f"exp_genpop_rest_3_{dataset_name}_{algo_name}"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  random_items = random.sample(range(exp.model.n_items), k=30)
  for i in random_items:
    args['item'] = i
    for blackbox in [1]:
      args['blackbox'] = blackbox
      name = f"[{i, blackbox}]"
      exp.run(f"{exp_folder}/{name}", Args(args))
      exp.model = load_model(dataset_name, algo_name, dataset)  

def single_user_log():

  dataset_name = 'Clothing_Shoes_and_Jewelry'
  algo_name = 'vbpr'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  random.seed(seed)

  args = {
    'seed': seed,
    'experiment': 'single_user',
    'dataset_name': dataset_name,
    'steps': 30,
    'epsilon': 1/255,
    'gamma': 7,
    'by_rank': 1,
    'blackbox': 1,
    'examples': 64,
  }

  args.update(default_args)

  exp_folder = f"exp_from_rank_range"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  random_users = random.sample(range(exp.model.n_users), k=150)
  random_items = random.sample(range(exp.model.n_items), k=150)
  for u, i in zip(random_users, random_items):
    args['user'] = u
    args['item'] = i
    name = f"[{u, i}]"
    exp.run(f"{exp_folder}/{name}", Args(args))

def main():
  # run_main_table_exp()
  # run_bb_examples_per_step_exp()
  # run_segment_exp()
  # run_from_rank_exp()
  # run_general_pop_exp()
  # single_user_log()
  # run_bb_single_user_restrict_exp()
  # run_bb_segment_restrict_exp()
  run_general_pop_restrict_exp()

if __name__ == '__main__':
  main()