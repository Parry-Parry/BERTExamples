import pyterrier as pt 
if not pt.started():
    pt.init()
import pandas as pd
import ir_datasets as irds
from transformers import AutoTokenizer, AutoModel
from pyterrier_dr import HgfBiEncoder, NumpyIndex
from fire import Fire

def embed_queries(dataset : str, out_path : str, subset : int = 0, model : str = "bert-base-uncased", batch_size : int = 256, lookup="msmarco-passage/train/triples-small", cutoff : int = 10):
    index = NumpyIndex(out_path)
    docpair_lookup = pd.DataFrame(irds.load(lookup).docpairs_iter()).drop_duplicates("query_id").set_index("query_id").doc_id_a.to_dict()
    queries = pd.DataFrame(irds.load(dataset).queries_iter()).rename(columns={"query_id":"qid", "text" : "query"})
   
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    hgf = HgfBiEncoder(model, tokenizer, batch_size=batch_size)

    pipe = hgf >> index % cutoff
    res = pipe.transform(queries).rename(columns={"qid":"query_id"})
    res['doc_id_a'] = res.docno.apply(lambda x : docpair_lookup[x])
    res[['query_id', 'doc_id_a']].to_json(out_path + ".jsonl", orient="records", lines=True)
    return "Done!"

if __name__ == "__main__":
    Fire(embed_queries)