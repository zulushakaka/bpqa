import gensim
import json
from collections import OrderedDict
# import h5py
import pickle
import os
import numpy as np


def load_word_embedding():
    embedding_path = 'data/webq_word_embedding.dump'
    if os.path.exists(embedding_path):
        with open(embedding_path, 'r') as f:
            word2idx, embedding = pickle.load(f)
            return word2idx, embedding
    print('Generating word embedding ...')
    vocabulary = get_webq_vocabulary()
    voc_size = len(vocabulary)
    print('vocabulary size: %d' % voc_size)
    word2idx = OrderedDict()
    embedding = np.zeros(dtype=np.float32, shape=(voc_size, 300))

    path = 'data/GoogleNews-vectors-negative300.bin'
    print('Loading raw word2vec data ...')
    model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    print('Raw data loaded.')

    idx = 0
    for word in vocabulary:
        word2idx[word] = idx
        if word in model:
            embedding[idx, :] = model[word]
        idx += 1

    with open(embedding_path, 'w') as f:
        pickle.dump((word2idx, embedding), f)

    return word2idx, embedding


def get_webq_vocabulary():
    vocab = set()

    def helper(path, voc):
        with open(path, 'r') as f:
            webq = json.load(f)
            for q in webq['Questions']:
                raw = q['RawQuestion']
                words = raw[:-1].split()
                for word in words:
                    voc.add(word)

    helper('data/WebQSP/WebQSP.train.json', vocab)
    helper('data/WebQSP/WebQSP.test.json', vocab)
    return vocab


if __name__ == '__main__':
    load_word_embedding()