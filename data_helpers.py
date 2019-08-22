import torch
import fastText
import os
import numpy as np

from config import EMBEDDING_DIM

class FastTextNN:
    def __init__(self, ft_model, ft_matrix=None):
        self.ft_model = ft_model        
        self.ft_words = ft_model.get_words()
        self.ft_matrix = ft_matrix
        if self.ft_matrix is None:
            self.ft_matrix = np.empty((len(self.ft_words), ft_model.get_dimension()))
            for i, word in enumerate(self.ft_words):
                self.ft_matrix[i,:] = ft_model.get_word_vector(word)

    def find_nearest_neighbor_vector(self, query, n=10, cossims=None):
        if cossims is None:
            cossims = np.matmul(self.ft_matrix, query, out=cossims)
        norms = np.sqrt((query**2).sum() * (self.ft_matrix**2).sum(axis=1))
        cossims = cossims/norms
        result_i = np.argpartition(-cossims, range(n+1))[0:n]
        indices = zip(result_i, cossims[result_i])
        return [(self.ft_words[r[0]], r[1]) for r in indices]
    
    def find_nearest_neighbor_word(self, query_word, n=10, cossims=None):
        query = self.ft_model.get_word_vector(query_word)
        res = self.find_nearest_neighbor_vector(query, n+1, cossims)
        if query_word in self.ft_words:
            res = res[1:n+1]
        else:
            res = res[0:n]
        return res

def load_embeddings_and_mapping(language_code):
    model = fastText.load_model(os.path.join(
        'data', 'fastText', f'wiki.{language_code}.bin'
    ))

    if language_code == 'en':
        mapping = None
    else:
        mapping = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        best_mapping_path = os.path.join(
            'data', 'fastText', f'best_mapping.{language_code}.pth')
        mapping_weights = torch.from_numpy(torch.load(best_mapping_path))
        mapping.weight.data.copy_(mapping_weights.type_as(mapping.weight.data))

    return model, mapping

def get_word_vector(model, mapping, word):
    vector = model.get_word_vector(word)
    if mapping:
        vector = torch.from_numpy(vector).type(torch.FloatTensor)
        vector = mapping(vector).data.numpy()
    return vector

if __name__ == '__main__':
    model, mapping = load_embeddings_and_mapping('en')
    nn = FastTextNN(model)