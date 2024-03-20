from typing import Any, Optional
import ir_datasets as irds
import pandas as pd

class ExampleStore(object):
    def __init__(self, ir_dataset : str, file : Optional[str] = None):

        ds = irds.load(ir_dataset)
        docs = pd.DataFrame(ds.docs_iter()).set_index("doc_id").text.to_dict()
        queries = pd.DataFrame(ds.queries_iter()).set_index("query_id").text.to_dict()

        if file is not None: 
            lookup = pd.read_json(file, orient="records", lines=True)
            lookup['rel_query'] = lookup['rel_query_id'].apply(lambda x : queries[x])
            lookup['pos_doc'] = lookup['doc_id_a'].apply(lambda x : docs[x])
            lookup['neg_doc'] = lookup['doc_id_b'].apply(lambda x : docs[x])
            self.lookup = lookup
            self.lookup_func = self._local
        else:
            ds = irds.load(ir_dataset)
            data = pd.DataFrame(ds.docpairs_iter())
            docs = pd.DataFrame(ds.docs_iter()).set_index("doc_id").text.to_dict()
            queries = pd.DataFrame(ds.queries_iter()).set_index("query_id").text.to_dict()
            data['query'] = data['query_id'].apply(lambda x : queries[x])
            data['pos_doc'] = data['doc_id_a'].apply(lambda x : docs[x])
            data['neg_doc'] = data['doc_id_b'].apply(lambda x : docs[x])
            self.lookup = data 
            self.lookup_func = self._random

    def _random(self, n : int, qid : Optional[str] = None):
        return [(row.query, row.pos_doc, row.neg_doc) for row in self.lookup.sample(n).itertuples()]

    def _local(self, n : int, qid : str):
        _subset = self.lookup[self.lookup['query_id'] == qid].iloc[:n]
        return [(row.query, row.pos_doc, row.neg_doc) for row in _subset.itertuples()]
    
    def __call__(self, n : int, qid : Optional[str] = None) -> Any:
        return self.lookup_func(n, qid)