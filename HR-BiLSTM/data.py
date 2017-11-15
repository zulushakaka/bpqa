import gensim
import json
from collections import OrderedDict
# import h5py
import pickle
import os
import numpy as np
import random


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
    embedding = np.zeros(dtype=np.float32, shape=(10000, 300))

    path = 'data/GoogleNews-vectors-negative300.bin'
    print('Loading raw word2vec data ...')
    model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    print('Raw data loaded.')

    idx = 1
    for word in vocabulary:
        word2idx[word] = idx
        if word in model:
            embedding[idx, :] = model[word]
        idx += 1

    for rel in get_fb_relations():
        rel_words = rel.replace('_', '.').split('.')
        for word in rel_words:
            if not word in word2idx:
                word2idx[word] = idx
                if word in model:
                    embedding[idx, :] = model[word]
                idx += 1

    with open(embedding_path, 'w') as f:
        pickle.dump((word2idx, embedding[:idx+1]), f)

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
                inf_chain = q['Parses'][0]['InferentialChain']
                if inf_chain:
                    for rel in inf_chain:
                        rel_words = rel.replace('_', '.').split('.')
                        for word in rel_words:
                            voc.add(word)

    helper('data/WebQSP/WebQSP.train.json', vocab)
    helper('data/WebQSP/WebQSP.test.json', vocab)
    return vocab


def get_fb_relations():
    rel2idx = OrderedDict()
    cnt = 1
    with open('data/fb_relations.txt', 'r') as f:
        for line in f:
            if line.startswith('http://rdf.freebase.com/ns/'):
                rel2idx[line[27:-1]] = cnt
                cnt += 1

    return rel2idx


def prepare_train_data(word2idx, rel2idx, max_q, max_r_r, max_r_w, batch):
    wq = []
    wq_len = []
    wr_rel = []
    wr_rel_len = []
    wr_word = []
    wr_word_len = []

    idx2rel = {}
    for k, v in rel2idx.items():
        idx2rel[v] = k

    with open('data/WebQSP/WebQSP.train.json', 'r') as f:
        webq = json.load(f)
        for q in webq['Questions']:
            raw = q['RawQuestion']
            words = map(lambda x: word2idx[x], raw[:-1].split())
            # sparql = q['Parses'][0]['Sparql']
            infChain = q['Parses'][0]['InferentialChain']
            if infChain == None:
                continue
            for rel in infChain:
                if not rel in rel2idx:
                    rel2idx[rel] = len(rel2idx) + 1
                    idx2rel[len(rel2idx)] = rel
            infChain_rel = map(lambda x: rel2idx[x], infChain)
            # infChain_word = map(lambda x: word2idx[x], sum([rel.split('.') for rel in infChain], []))
            for rel in infChain_rel:
                batch_wq = []
                batch_wq_len = []
                batch_wr_rel = []
                batch_wr_rel_len = []
                batch_wr_word = []
                batch_wr_word_len = []
                batch_wq.append(words + [0 for i in range(max_q - len(words))])
                batch_wq_len.append(len(words))
                batch_wr_rel.append([rel] + [0 for i in range(max_r_r - 1)])
                batch_wr_rel_len.append(1)
                infChain_word = map(lambda x: word2idx[x], idx2rel[rel].replace('_', '.').split('.'))
                batch_wr_word.append(infChain_word + [0 for i in range(max_r_w - len(infChain_word))])
                batch_wr_word_len.append(len(infChain_word))
                negative = []
                for _ in range(batch - 1):
                    neg = random.randint(1, len(rel2idx))
                    while neg == rel or neg in negative:
                        neg = random.randint(1, len(rel2idx))
                    negative.append(neg)

                    batch_wq.append(words + [0 for i in range(max_q - len(words))])
                    batch_wq_len.append(len(words))
                    batch_wr_rel.append([neg] + [0 for i in range(max_r_r - 1)])
                    batch_wr_rel_len.append(1)
                    neg_words = map(lambda x: word2idx[x], idx2rel[neg].replace('_', '.').split('.'))
                    batch_wr_word.append(neg_words + [0 for i in range(max_r_w - len(neg_words))])
                    batch_wr_word_len.append(len(neg_words))
                wq.append(batch_wq)
                wq_len.append(batch_wq_len)
                wr_rel.append(batch_wr_rel)
                wr_rel_len.append(batch_wr_rel_len)
                wr_word.append(batch_wr_word)
                wr_word_len.append(batch_wr_word_len)

    return np.asarray(wq), np.asarray(wq_len), np.asarray(wr_rel), \
           np.asarray(wr_rel_len), np.asarray(wr_word), np.asarray(wr_word_len)

if __name__ == '__main__':
    load_word_embedding()