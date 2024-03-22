import pyterrier as pt
pt.init()
import fire 
import os
from os.path import join
import ir_datasets as irds
import pandas as pd
from ..modelling.prp import PRP
from ..examples import ExampleStore
import random
import numpy as np
import torch
import json

def run(topics_or_res : str, 
         out_dir : str, 
         model_name_or_path : str,
         batch_size : int = 4,
         window_size : int = None,
         few_shot_mode : str = 'random',
         score_func : str = 'allpair',
         k : int = 0,
         n_pass : int = 10,
         k_shot_file : str = None,
         eval : str = 'msmarco-passage/trec-dl-2019/judged',
         dataset : str = 'irds:msmarco-passage/train/triples-small',
         seed : int = 42,
         ):  
    
    print(k)
    print(k_shot_file)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    lookup = irds.load(eval)
    queries = pd.DataFrame(lookup.queries_iter()).set_index('query_id').text.to_dict()
    dataset = pt.get_dataset(dataset)
    topics_or_res = pt.io.read_results(topics_or_res)
    topics_or_res['query'] = topics_or_res['qid'].apply(lambda x: queries[str(x)])
    topics_or_res = pt.text.get_text(dataset, "text")(topics_or_res)
    del queries

    os.makedirs(out_dir, exist_ok=True)

    if k_shot_file:
        sim = "bert" if "bert" in k_shot_file else "bm25"

    eval_set = "19" if "2019" in eval else "20"
    out = join(out_dir, f"{eval_set}.{k}.{few_shot_mode}.{score_func}.{sim}.{seed}")

    if os.path.exists(f"{out}.res.gz"): return "Already exists"

    if k > 0: 
        few_shot_examples = ExampleStore(eval.replace('irds:', ''), file=k_shot_file)
    else: few_shot_examples = None

    try: 
        model = PRP(model_name_or_path, 
                     batch_size=batch_size, 
                     k=k, 
                     window_size=window_size, 
                     n_pass=n_pass, 
                     few_shot_mode=few_shot_mode, 
                     score_func=score_func,
                     few_shot_examples=few_shot_examples)
    except OSError as e: return f"Failed to load model, {e}"

    res, log = model.transform(topics_or_res)
    with open(join(out_dir, f"{out}.log.json"), 'w') as f:
        json.dump(log, f)
    pt.io.write_results(res, f"{out}.res.gz")


    return "Success!"

if __name__ == '__main__':
    fire.Fire(run) 