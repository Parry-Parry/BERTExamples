
from collections import defaultdict
import pyterrier as pt
if not pt.started():
    pt.init()
import pandas as pd
import torch 
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import numpy as np
from more_itertools import chunked
from .prompt import create_prompt
from pyterrier.model import add_ranks
import random
from . import _iter_windows

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
                 few_shot_mode : str = 'random',
                 score_func : str = 'allpair',
                 few_shot_examples : Optional[list] = None,
                 k : int = 1,
                 n : int = 10,
                 window_size : int = None,
                 n_pass : int = 3):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.max_len = self.tokenizer.model_max_length
        self.batch_size = batch_size
        self.score_func = {
            'allpair': self._all_pair,
            'sliding_window': self._sliding_window,
            'partial_bubble' : self._partial_bubble_sort
        }[score_func]
        self.few_shot_func = self._random if few_shot_mode == 'random' else self._topk
        self.few_shot_examples = few_shot_examples
        self.k = k
        self.n = n
        self.window_size = window_size
        self.n_pass = n_pass

        self.A = self.tokenizer.encode("1", return_tensors="pt", add_special_tokens=False)[0][1] # remove or add [1] based on different tokens
        self.B = self.tokenizer.encode("2", return_tensors="pt", add_special_tokens=False)[0][1] # remove or add [1] based on different tokens
    
    def _random(self, qid : str):
        if self.few_shot_examples is None: return None
        else: return random.sample(self.few_shot_examples(self.n, qid), self.k)
    
    def _topk(self, qid : str):
        if self.few_shot_examples is None: return None
        else: return self.few_shot_examples(self.n, qid)[:self.k]

    def _score_set(self, qid : str, query : str, doc_texts : list, doc_ids : list):
        idx = create_pairs(len(doc_texts))
        score_matrix = np.zeros((len(doc_texts), len(doc_texts)))

        for batch in tqdm(chunked(idx, self.batch_size), unit='batch'):
            prompts = [create_prompt(query, doc_texts[i], doc_texts[j], self.few_shot_func(qid)) for i, j in batch]
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len, return_special_tokens_mask=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(**inputs, do_sample=False, temperature=0.0, top_p=None, return_dict_in_generate=True, output_scores=True, max_new_tokens=1).scores[0]
            scores = outputs[:, (self.A, self.B)].softmax(dim=-1)[:, 0].tolist()
            for (i, j), score in zip(batch, scores):
                score_matrix[i, j] = score
        
        for i in range(len(doc_texts)):
            for j in range(len(doc_texts)):
                if i == j: score_matrix[i, j], score_matrix[j, i] = 0., 0.
                elif score_matrix[i, j] > 0.5 and score_matrix[j, i] < 0.5: score_matrix[i, j], score_matrix[j, i] = 1., 0.
                elif score_matrix[i, j] < 0.5 and score_matrix[j, i] > 0.5: score_matrix[i, j], score_matrix[j, i] = 0., 1.
                else: score_matrix[i, j], score_matrix[j, i] = 0.5, 0.5

        log = defaultdict(dict)
        for i in range(len(doc_texts)):
            log[qid][doc_ids[i]] = {doc_ids[j] : item for j, item in enumerate(score_matrix[i].tolist())}
        
        scores = np.sum(score_matrix, axis=1)
        return np.argsort(scores), log
    
    def _all_pair(self, qid : str, query : str, query_results : pd.DataFrame):
        doc_texts = query_results['text'].to_numpy()
        doc_ids = query_results['docid'].to_numpy()
        
        order, log = self._score_set(qid, query, doc_ids.tolist(), doc_texts.tolist())
        scores = [1.0/(i+1) for i in range(len(order))]
        doc_texts = doc_texts[order]
        doc_ids = doc_ids[order]

        return doc_ids, doc_texts, scores, log

    def _sliding_window(self, qid : str, query : str, query_results : pd.DataFrame):
        doc_texts = query_results['text'].to_numpy()
        doc_ids = query_results['docid'].to_numpy()

        window_size = self.window_size
        stride = window_size / 2
        log = {}
        for start_idx, end_idx, window_len in _iter_windows(len(doc_texts), window_size, stride):
            order, _log = self._score_set(qid, query, doc_ids[start_idx:end_idx].tolist(), doc_texts[start_idx:end_idx].tolist())
            doc_texts[start_idx:end_idx] = doc_texts[order]
            doc_ids[start_idx:end_idx] = doc_ids[order]
            log.update(_log)

        scores = [1.0/(i+1) for i in range(len(doc_texts))]
        return doc_ids, doc_texts, scores, log
    
    def _partial_bubble_sort(self, qid : str, query : str, query_results : pd.DataFrame):
        doc_texts = query_results['text'].to_numpy()
        doc_ids = query_results['docid'].to_numpy()

        log = defaultdict(dict)
        for i in range(len(doc_texts)):
            if i > self.n_pass: break
            idx = len(doc_texts) - i
            for j in range(idx, 1, -1): # do pairwise swaps from idx to the start
                doc_one = doc_texts[j]
                doc_two = doc_texts[j-1]
                prompts = [create_prompt(query, doc_one, doc_two, self.few_shot_func(qid)), create_prompt(query, doc_two, doc_one, self.few_shot_func(qid))]

                inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len, return_special_tokens_mask=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model.generate(**inputs, do_sample=False, temperature=0.0, top_p=None, return_dict_in_generate=True, output_scores=True, max_new_tokens=1).scores
                scores = outputs[0][:, (self.A, self.B)].softmax(dim=-1)

                if scores[0, 0] > scores[0, 1] and scores[1, 1] > scores[1, 0]:
                    doc_texts[j], doc_texts[j-1] = doc_texts[j-1], doc_texts[j]
                    doc_ids[j], doc_ids[j-1] = doc_ids[j-1], doc_ids[j]
                    log[qid][doc_ids[j]][doc_ids[j-1]] = 1.
                    log[qid][doc_ids[j-1]][doc_ids[j]] = 0.
                elif scores[0, 0] < scores[0, 1] and scores[1, 1] < scores[1, 0]:
                    log[qid][doc_ids[j]][doc_ids[j-1]] = 0.
                    log[qid][doc_ids[j-1]][doc_ids[j]] = 1.
                else:
                    log[qid][doc_ids[j]][doc_ids[j-1]] = 0.5
                    log[qid][doc_ids[j-1]][doc_ids[j]] = 0.5
        
        scores = [1.0/(i+1) for i in range(len(doc_texts))]
        return doc_ids, doc_texts, scores, log

    def transform(self, topics_or_res : pd.DataFrame, few_shot_examples : Optional[list] = None):
        res = {
            'qid': [],
            'query': [],
            'docno': [],
            'text': [],
            'score': [],
        }

        log = {}
        with torch.no_grad():
            for (qid, query), query_results in tqdm(topics_or_res.groupby(['qid', 'query']), unit='q'):
                # for _all-pair processing:
                doc_ids, doc_texts, scores, _log = self.score_func(qid, query, query_results, few_shot_examples)
                res['qid'].extend([qid] * len(query_results))
                res['query'].extend([query] * len(query_results))
                res['docno'].extend(doc_ids)
                res['text'].extend(doc_texts)
                res['score'].extend(scores)

                log.update(_log)

        res = pd.DataFrame(res)
        return add_ranks(res), log