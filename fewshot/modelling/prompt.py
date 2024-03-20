zero_shot = """
<|system|>
You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query</s> 
<|user|>
which of the following two passages is more relevant to the query "{query}"?
Passage 1 : {doc_one}
Passage 2 : {doc_two}
Only respond with either 1 or 2, do not say any word or explain.</s> 
<|assistant|>
"""

shot_constructor = """
Query: {related_query}
Passage 1 : {rel_doc}
Passage 2 : {nrel_doc}
Output: 1
            
####
"""

few_shot = """
<|system|>
You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query</s> 
<|user|>
which of the following two passages is more relevant to the query? Only respond with either 1 or 2, do not say any word or explain.

{examples}

Query: {query}
Passage 1 : {doc_one}
Passage 2 : {doc_two}
Output: </s> 
<|assistant|>
"""

def create_prompt(
        query: str,
        doc_one: str,
        doc_two: str,
        few_shot_examples: list = None,
    ):
    if few_shot_examples:
        examples = "\n".join([shot_constructor.format(related_query=ex[0], rel_doc=ex[1], nrel_doc=ex[2]) for ex in few_shot_examples])
        return few_shot.format(query=query, doc_one=doc_one, doc_two=doc_two, examples=examples)
    else:
        return zero_shot.format(query=query, doc_one=doc_one, doc_two=doc_two)
    