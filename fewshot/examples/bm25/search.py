import pyterrier as pt 
if not pt.started():
    pt.init()
import pandas as pd
import ir_datasets as irds
from pyterrier_pisa import PisaIndex
from fire import Fire
import multiprocessing as mp

def search_queries(dataset : str, index_path : str, out_path : str, lookup="msmarco-passage/train/triples-small", cutoff : int = 10, k1 : int = 1.2, b : int = 0.75):
    index = PisaIndex(index_path, threads=mp.cpu_count()).bm25(k1=k1, b=b, num_results=cutoff)
    docpair_lookup = pd.DataFrame(irds.load(lookup).docpairs_iter()).drop_duplicates("query_id").set_index("query_id").doc_id_a.to_dict()
    queries = pd.DataFrame(irds.load(dataset).queries_iter()).rename(columns={"query_id":"qid", "text" : "query"})

    pipe = index % cutoff
    res = pipe.transform(queries).rename(columns={"qid":"query_id", "docno" : "rel_query_id"})[['query_id', 'rel_query_id']]
    res['rel_doc_id_a'] = res.rel_query_id.apply(lambda x : docpair_lookup[x])

    # group by query_id and make a list of rel_query_id, doc_id_a tuples
    out = res.groupby('query_id').apply(lambda x: list(zip(x['rel_query_id'], x['rel_doc_id_a']))).reset_index(name='rels')
    out.to_json(out_path + ".jsonl", orient="records", lines=True)
    return "Done!"

if __name__ == "__main__":
    Fire(search_queries)