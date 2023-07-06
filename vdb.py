import hnswlib
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer



class VDB:
    def __init__(
            self,
            space = 'cosine', 
            dim = 384,
            index_max_elements = 10000,
            model_name = 'sentence-transformers/all-MiniLM-L6-v2',
        ):

        index = hnswlib.Index(space = space, dim = dim)
        index.init_index(max_elements = index_max_elements)
        index.set_ef(50)
        self.index = index

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)

    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min = 1e-9
        )

    def __get_embeddings(self, data):
        encoded_input = self.tokenizer(data, padding = True, truncation = True, return_tensors = 'pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        sentence_embeddings = self.__mean_pooling(model_output, encoded_input["attention_mask"])
        return torch.nn.functional.normalize(sentence_embeddings, p = 2, dim = 1)

    def __quantize(self, embeddings):
        pass

    def write(self, data):
        embeddings = self.__get_embeddings(data)
        self.index.add_items(embeddings)

    def search(self, data, k = 5):
        embeddings = self.__get_embeddings(data)
        return self.index.knn_query(embeddings, k = k)
