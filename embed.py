import pyterrier as pt 
if not pt.started():
    pt.init()
import pandas as pd
import ir_datasets as irds
from pyterrier_dr import HgfBiEncoder, NumpyIndex
from fire import Fire

def embed_queries(out_path : str, subset : int = 0, model : str = "bert-base-uncased", batch_size : int = 256, dataset : str = "msmarco-passage/train/triples-small"):
    index = NumpyIndex(out_path)
    idx = pd.DataFrame(irds.load(dataset).docpairs_iter()).drop_duplicates("query_id").rename(columns={"query_id":"docno"})['docno'].to_list()
    queries = pd.DataFrame(irds.load(dataset).queries_iter()).rename(columns={"query_id":"docno"}).set_index("docno").text.to_dict()
    texts = [queries[qid] for qid in idx]
    frame = pd.DataFrame({"docno": idx, "text" : texts})

    if subset > 0:
        frame = frame.sample(n=subset)

    hgf = HgfBiEncoder.from_pretrained(model, batch_size=batch_size)

    pipe = hgf >> index
    pipe.index(frame)
    return "Done!"

if __name__ == "__main__":
    Fire(embed_queries)