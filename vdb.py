import hnswlib
import numpy as np
# from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

class VDB:
    def __init__(
            self,
            space = 'cosine', 
            dim = 384,
            index_max_elements = 10000,
            model_name = 'all-MiniLM-L6-v2',
        ):

        self.dimension = dim

        index = hnswlib.Index(space = space, dim = dim)
        index.init_index(max_elements = index_max_elements)
        index.set_ef(50)
        self.index = index
        
        self.model = SentenceTransformer(model_name)

       # self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
       # self.model = AutoModel.from_pretrained(model_name)


    def __quantize(self, embeddings):
        pass

    def write(self, data):
        query_embedding = self.model.encode(data)
        self.index.add_items(query_embedding)

    def search(self, data, k = 5):
        query_embedding = self.model.encode(data)
        return self.index.knn_query(query_embedding, k = k)
