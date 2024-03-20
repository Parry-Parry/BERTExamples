import pyterrier as pt 
if not pt.started():
    pt.init()
import pandas as pd
import ir_datasets as irds
from pyterrier_pisa import PisaIndex
from fire import Fire
from tqdm import tqdm

def index_queries(out_path : str, subset : int = 0, model : str = "bert-base-uncased", batch_size : int = 256, dataset : str = "msmarco-passage/train/triples-small"):
    index = PisaIndex(out_path)
    idx = pd.DataFrame(irds.load(dataset).docpairs_iter()).drop_duplicates("query_id").rename(columns={"query_id":"docno"})['docno'].to_list()
    queries = pd.DataFrame(irds.load(dataset).queries_iter()).rename(columns={"query_id":"docno"}).set_index("docno").text.to_dict()
    texts = [queries[qid] for qid in idx]
    frame = pd.DataFrame({"docno": idx, "text" : texts})

    if subset > 0:
        frame = frame.sample(n=subset)

    iterator = tqdm((row.to_dict() for index, row in frame.iterrows()), total=len(frame))

    index.index(iterator)
    return "Done!"

if __name__ == "__main__":
    Fire(index_queries)