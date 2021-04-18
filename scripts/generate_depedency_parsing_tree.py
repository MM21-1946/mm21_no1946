dataset_list = [
    ('activitynet1.3', 'train'), ('activitynet1.3', 'val'), ('activitynet1.3', 'test'), \
    ('charades_sta', 'train'), ('charades_sta', 'test'), \
    ('tacos', 'train'), ('tacos', 'val'), ('tacos', 'test')
]

import spacy
import json
from tqdm import tqdm as tqdm # notebook
from copy import deepcopy
import multiprocessing as mp

dep_rel2index = {}
def register_dep_rel(dep_rel2index, rel):
    if rel not in dep_rel2index:
        dep_rel2index[rel] = len(dep_rel2index) + 2 # 1 for self loop

def transform_dataset_single_replace(dataset, split, pos, lock):
    nlp = spacy.load("en_core_web_sm")
    with open(f'{dataset}/annotations/{split}.json', 'r') as f:
        data = json.load(f)
    # Transform
    with lock:
        pbar = tqdm(total=len(data), desc=f'Processing {dataset}-{split} set...', position=pos)
    for _, vid in enumerate(data):
        data[vid]['dependency_parsing_graph'] = []
        for k, sent in enumerate(data[vid]['sentences']):
            doc = nlp(sent)
            sen_len = len(doc)
            adj_mat = [[0]*sen_len for _ in range(sen_len)]
            for i, word in enumerate(doc):
                adj_mat[i][i] = 1
                children = list(word.children)
                if len(children):
                    for child in children:
                        dep_rel = child.dep_
                        register_dep_rel(dep_rel2index, dep_rel)
                        adj_mat[i][child.i] = dep_rel2index[dep_rel]
                data[vid]['dependency_parsing_graph'].append(adj_mat)
        with lock:
            pbar.update(1)
    with open('{}/annotations/{}.json'.format(dataset, split), 'w') as f:
            json.dump(data, f)
    with lock:
        pbar.close()

def transform_dataset(dataset_list):
    tqdm_lock = mp.Lock() # for tqdm thread-safe, see https://github.com/tqdm/tqdm/issues/407
    # Create tasks
    task_queue = []
    for i, (dataset, split) in enumerate(dataset_list):
        # Load original datasets
        task_queue.append(
            mp.Process(target=transform_dataset_single_replace, args=(dataset, split, i+1, tqdm_lock,))
        )
    for task in task_queue:
        task.start()
    for task in task_queue:
        task.join()

transform_dataset(dataset_list)