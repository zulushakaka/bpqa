import gensim


def load_word_embedding():
    path = '/home/xianyang/bpqa/data/GoogleNews-vectors-negative300.bin'
    model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    print(model.wv.most_similar(positive=['woman', 'king'], negative=['man']))


if __name__ == '__main__':
    load_word_embedding()