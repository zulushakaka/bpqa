import gensim


def load_word_embedding():
    path = '/home/xianyang/bpqa/data/GoogleNews-vectors-negative300.bin'
    print('Loading word2vec data ...')
    model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    sentence = ["london", "is", "the", "capital", "great", "britain"]
    vectors = [model[w] for w in sentence]
    print(model.most_similar(positive=['woman', 'king'], negative=['man']))
    # print(vectors)


if __name__ == '__main__':
    load_word_embedding()