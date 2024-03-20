import pyterrier as pt
if not pt.started():
    pt.init()
import pandas as pd
import torch 
from tqdm import tqdm
from transformers import AutoModelForCaus_allM, AutoTokenizer
from typing import Optional
import numpy as np
from more_itertools import chunked
from .prompt import create_prompt
from pyterrier.model import add_ranks
import random

def create_pairs(num : int):
    array = []
    for i in range(num):
        for j in range(num):
            array.append((i, j))
    return array
                
class PRP(pt.transformer):
    def __init__(self, 
                 model_name_or_path : str, 
                 batch_size : int = 4,
                 mode : str = 'all',
                 few_shot_mode : str = 'random',
                 k : int = 1,
                 window_size : int = 10,
                 n_pass : int = 3):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCaus_allM.from_pretrained(model_name_or_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.max_len = self.tokenizer.model_max_length
        self.batch_size = batch_size
        self.score_func = self._all if mode == 'all' else self._sliding_window
        self.few_shot_func = self._random if few_shot_mode == 'random' else self._topk
        self.k = k
        self.window_size = window_size
        self.n_pass = n_pass

        self.A = self.tokenizer.encode("1", return_tensors="pt", add_special_tokens=False)[0][1] # remove or add [1] based on different tokens
        self.B = self.tokenizer.encode("2", return_tensors="pt", add_special_tokens=False)[0][1] # remove or add [1] based on different tokens
    
    def _random(self, few_shot_examples : Optional[list] = None):
        if few_shot_examples is None:
            return None
        else:
            return random.sample(few_shot_examples, self.k)
    
    def _topk(self, few_shot_examples : Optional[list] = None):
        if few_shot_examples is None:
            return None
        else:
            return few_shot_examples[:self.k]

    def _all(self, qid : str, query : str, query_results : pd.DataFrame, few_shot_examples : Optional[list] = None):
        doc_texts = query_results['text'].tolist()
        doc_ids = query_results['docid'].tolist()
        score_matrix = np.zeros((len(doc_texts), len(doc_texts)))

        pair_idx = create_pairs(len(doc_texts))

        for batch in tqdm(chunked(pair_idx, self.batch_size), unit='batch'):
            prompts = [create_prompt(query, doc_texts[i], doc_texts[j], few_shot_examples) for i, j in batch]
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len, return_special_tokens_mask=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(**inputs, do_sample=False, temperature=0.0, top_p=None, return_dict_in_generate=True, output_scores=True, max_new_tokens=1).scores[0]
            scores = outputs[:, (self.A, self.B)].softmax(dim=-1)[:, 0].tolist()
            for (i, j), score in zip(batch, scores):
                score_matrix[i, j] = score
        
        for i in range(len(doc_texts)):
            for j in range(len(doc_texts)):
                if i == j:
                    score_matrix[i, j] = 0.
                    score_matrix[j, i] = 0.
                elif score_matrix[i, j] > 0.5 and score_matrix[j, i] < 0.5:
                    score_matrix[i, j] = 1.
                    score_matrix[j, i] = 0.
                elif score_matrix[i, j] < 0.5 and score_matrix[j, i] > 0.5:
                    score_matrix[i, j] = 0.
                    score_matrix[j, i] = 1.
                else:
                    score_matrix[i, j] = 0.5
                    score_matrix[j, i] = 0.5

        scores = np.sum(score_matrix, axis=1).tolist()

        # convert to dict lookup based on score matrix correlating with row idx
        log = {}
        for i in range(len(doc_texts)):
            log[doc_ids[i]] = {doc_ids[j] : item for j, item in enumerate(score_matrix[i].tolist())}
            
        return scores, log

    def _sliding_window(self, qid : str, query : str, query_results : pd.DataFrame, few_shot_examples : Optional[list] = None):
        doc_texts = query_results['text'].tolist()
        doc_ids = query_results['docid'].tolist()

        num_pass = self.n_pass
        ranks = []

        n = len(doc_texts)
        w = self.window_size

        for i in range(n):
            ranks.append(i+1)
            if i < num_pass:
                swapped = False
                for j in range(0, n-i-1):
                    for k in range(1, w):
                        if n-j-w < 0:
                            break
                        doc_one = doc_texts[n-j-w+k]
                        doc_two = doc_texts[n-j-w]
    
                        # Generate prompts
                        prompts = [create_prompt(query, doc_one, doc_two, few_shot_examples), create_prompt(query, doc_two, doc_one, few_shot_examples)]
                        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len, return_special_tokens_mask=True)
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                        outputs = self.model.generate(**inputs, do_sample=False, temperature=0.0, top_p=None, return_dict_in_generate=True, output_scores=True, max_new_tokens=1).scores
                        a_scores = outputs[0][:, (self.A, self.B)].softmax(dim=-1)[:, 0].tolist()
                        b_scores = outputs[1][:, (self.A, self.B)].softmax(dim=-1)[:, 0].tolist()
           
                        # Check if a swap is needed and perform if so
                        if a_scores[0] > b_scores[0] and b_scores[1] > b_scores[0]:
                            doc_texts[n-j-w+k], doc_texts[n-j-w] = doc_texts[n-j-w], doc_texts[n-j-w+k]
                            doc_ids[n-j-w+k], doc_ids[n-j-w] = doc_ids[n-j-w], doc_ids[n-j-w+k]
                            swapped = True

        # assign scores based on current position
        scores = [1.0/rank for rank in ranks]

        return scores, None

    def transform(self, topics_or_res : pd.DataFrame, few_shot_examples : Optional[list] = None):
        res = {
            'qid': [],
            'query': [],
            'docno': [],
            'text': [],
            'score': [],
        }

        prediction_logs = {}
        with torch.no_grad():
            for (qid, query), query_results in tqdm(topics_or_res.groupby(['qid', 'query']), unit='q'):
                # for _all-pair processing:
                scores, log = self.score_func(qid, query, query_results, few_shot_examples)
                res['qid'].extend([qid] * len(query_results))
                res['query'].extend([query] * len(query_results))
                res['docno'].extend(query_results['docno'].tolist())
                res['text'].extend(query_results['text'].tolist())
                res['score'].extend(scores)
                if log is not None:
                    prediction_logs[qid] = log
        res = pd.DataFrame(res)
        # sort by qid and scores 
        res = res.sort_values(['qid', 'score'], ascending=[True, False])
        return add_ranks(res), prediction_logs