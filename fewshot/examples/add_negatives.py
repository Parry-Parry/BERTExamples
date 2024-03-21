import pyterrier as pt 
if not pt.started():
    pt.init()
import pandas as pd
import ir_datasets as irds
from pyterrier_pisa import PisaIndex
from fire import Fire
import multiprocessing as mp
import json

SEED = 42

def add_negatives(examples_path : str, out_path : str = None, lookup="msmarco-passage/train/triples-small", cutoff : int = 100, k1 : int = 1.2, b : int = 0.75):
    index = PisaIndex.from_dataset("msmarco-passage", threads=mp.cpu_count()).bm25(k1=k1, b=b, num_results=cutoff)
    queries = pd.DataFrame(irds.load(lookup).queries_iter()).rename(columns={"query_id":"qid", "text" : "query"}).set_index("qid").query.to_dict()
    
    with open(examples_path) as f:
        examples = json.load(f)

    new_set = []
    for query in examples:
        record = {
            'query_id': query['query_id'],
            'examples': []
        }
        for rel_query_id, rel_doc_id_a in query['examples']:
            res = index.search(queries[rel_query_id])
            # filter to ranks 50-100
            negs = res[res["rank"] >= 50]
            if len(negs) == 0:
                neg = res.sample(1, seed=SEED)
            else:
                neg = negs.sample(1, seed=SEED)
            record['examples'].append({
                'rel_query_id' : rel_query_id,
                'rel_doc_id_a' : rel_doc_id_a,
                'rel_doc_id_b' : neg['docno'].values[0]
            })

    if out_path is None:
        out_path = examples_path
    with open(out_path, 'w') as f:
        json.dump(new_set, f)
   
    return "Done!"

if __name__ == "__main__":
    Fire(add_negatives)