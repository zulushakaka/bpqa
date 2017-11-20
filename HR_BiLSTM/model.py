import tensorflow as tf
import numpy as np
from data import *


LSTM_HIDDEN_SIZE = 100
MAX_QUESTION_LENGTH = 30
MAX_RELATION_WORD_LEGNTH = 10
MAX_RELATION_TYPE_LENGTH = 1
REL_EMBEDDING_SIZE = 300
MARGIN = 0.5
BATCH_SIZE = 16
LEARNING_RATE = 0.5


class HRBiLSTM (object):
    def __init__(self):
        self.word2idx, embedding_matrix = load_word_embedding()
        self.rel2idx = get_fb_relations()

        self.similarity, self.loss, self.q_inputs, self.q_length,\
        self.r_inputs_word, self.r_inputs_word_len, self.r_inputs_rels, self.r_inputs_rels_len =\
        self.build_model(embedding_matrix)
        self.sess = None

    def build_model(self, embedding_matrix):
        # question representation

        q_inputs = tf.placeholder(dtype=tf.int32, shape=[None, MAX_QUESTION_LENGTH])
        # dimension: (batch_size, max_seq_len)
        q_length = tf.placeholder(dtype=tf.int32, shape=[None])
        # dimension: (batch_size)

        embedding = tf.get_variable(name="embedding", shape=embedding_matrix.shape,
                                         initializer=tf.constant_initializer(embedding_matrix), trainable=False)
        # dimension: (voc_size, embedding_dim)
        embedded = tf.nn.embedding_lookup(embedding, q_inputs)
        # dimension: (batch_size, max_seq_len, embedding_dim)

        with tf.variable_scope('layer1') as scope:
            cell_q1_fw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
            cell_q1_bw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
            q1_outputs, _ = tf.nn.bidirectional_dynamic_rnn\
                (cell_fw=cell_q1_fw, cell_bw=cell_q1_bw, inputs=embedded, sequence_length=q_length, dtype=tf.float32)
        q1_outputs = tf.concat(q1_outputs, 2)
        # dimension: (batch_size, max_seq_len, cell_out_size * 2)

        with tf.variable_scope('layer2') as scope:
            cell_q2_fw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
            cell_q2_bw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
            q2_outputs, _ = tf.nn.bidirectional_dynamic_rnn\
                (cell_fw=cell_q2_fw, cell_bw=cell_q2_bw, inputs=q1_outputs, sequence_length=q_length, dtype=tf.float32)
        q2_outputs = tf.concat(q2_outputs, 2)
        # dimension: (batch_size, max_seq_len, cell_out_size * 2)

        q_outputs = tf.add(q1_outputs, q2_outputs)
        # dimension: (batch_size, max_seq_len, cell_out_size * 2)

        q_max_pool = tf.reduce_max(q_outputs, 1)
        # dimension: (batch_size, cell_out_size* 2)

        # relation representation

        r_inputs_word = tf.placeholder(tf.int32, shape=[None, MAX_RELATION_WORD_LEGNTH])
        # dimension: (batch_size, max_seq_len)
        r_inputs_word_len = tf.placeholder(tf.int32, shape=[None])
        r_inputs_rels = tf.placeholder(tf.int32, shape=[None, MAX_RELATION_TYPE_LENGTH])
        # dimension: (batch_size, max_seq_len)
        r_inputs_rels_len = tf.placeholder(tf.int32, shape=[None])

        rel_embedding_shape = [len(self.rel2idx), REL_EMBEDDING_SIZE]
        r_embedding = tf.get_variable(name="r_embedding", shape=rel_embedding_shape,
                    initializer=tf.random_normal_initializer(dtype=tf.float32), trainable=True)
        # relation embeddings are randomly initialized
        # dimension: (rel_voc_size, rel_embedding_dim)

        r_word_embedded = tf.nn.embedding_lookup(embedding, r_inputs_word)
        # dimension: (batch_size, max_seq_len, embedding_dim)
        r_rels_embedded = tf.nn.embedding_lookup(r_embedding, r_inputs_rels)
        # dimension: (batch_size, max_seq_len, rel_embedding_dim)

        with tf.variable_scope('rel') as scope:
            cell_r_fw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
            cell_r_bw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)

            r_word_outputs, _ = tf.nn.bidirectional_dynamic_rnn\
                (cell_fw=cell_r_fw, cell_bw=cell_r_bw,
                 inputs=r_word_embedded, sequence_length=r_inputs_word_len, dtype=tf.float32)
            # dimension: (batch_size, max_seq_len, cell_out_size) * 2
            r_rels_outputs, _ = tf.nn.bidirectional_dynamic_rnn\
                (cell_fw=cell_r_fw, cell_bw=cell_r_bw,
                 inputs=r_rels_embedded, sequence_length=r_inputs_rels_len, dtype=tf.float32)
            # dimension: (batch_size, max_seq_len, cell_out_size) * 2

        r_outputs = tf.concat([tf.concat(r_word_outputs, 2), tf.concat(r_rels_outputs, 2)], axis=1)
        # dimension: (batch_size, max_seq_len_word + max_seq_len_rel, cell_out_size * 2)
        r_max_pool = tf.reduce_max(r_outputs, 1)
        # dimension: (batch_size, cell_out_size * 2)

        similarity = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(q_max_pool, dim=1),
                                          tf.nn.l2_normalize(r_max_pool, dim=1)), axis=1)
        # dimension: (batch_size)

        # compute ranking loss
        s_true = similarity[0]
        s_max = tf.reduce_max(similarity[1:])
        loss = tf.maximum(0.0, MARGIN - s_true + s_max)

        return similarity, loss, q_inputs, q_length, r_inputs_word, r_inputs_word_len, r_inputs_rels, r_inputs_rels_len

    def train(self, epoch, prev=None):
        q, q_len, rr, rr_len, rw, rw_len = \
            prepare_train_data(self.word2idx, self.rel2idx, MAX_QUESTION_LENGTH, MAX_RELATION_TYPE_LENGTH,
                               MAX_RELATION_WORD_LEGNTH, BATCH_SIZE)
        print q.shape, q_len.shape, rr.shape, rr_len.shape, rw.shape, rw_len.shape

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            num_example = q.shape[0]

            if prev:
                saver.restore(sess, prev)
                # print('Model restored from file: %s' % load_path)
                prev_epoch = int(prev[prev.find('-') + 1: prev.find('.')])
            else:
                prev_epoch = 0

            for _ in range(epoch):
                # train
                for eid in range(num_example):
                    sess.run(optimizer, feed_dict={self.q_inputs: q[eid], self.q_length: q_len[eid],
                                                   self.r_inputs_word: rw[eid], self.r_inputs_word_len: rw_len[eid],
                                                   self.r_inputs_rels: rr[eid], self.r_inputs_rels_len: rr_len[eid]})
                # show eval
                correct = 0
                for i in range(num_example/100):
                    eid = random.randint(0, num_example-1)
                    sim = sess.run([self.similarity],
                                   feed_dict={self.q_inputs: q[eid], self.q_length: q_len[eid],
                                              self.r_inputs_word: rw[eid], self.r_inputs_word_len: rw_len[eid],
                                              self.r_inputs_rels: rr[eid], self.r_inputs_rels_len: rr_len[eid]})
                    if np.argmax(sim) == 0:
                        correct += 1
                # print '.'
                print correct,'/',num_example/100, float(correct)/num_example*100

            correct = 0
            for i in range(num_example):
                eid = i
                sim = sess.run([self.similarity],
                               feed_dict={self.q_inputs: q[eid], self.q_length: q_len[eid],
                                          self.r_inputs_word: rw[eid], self.r_inputs_word_len: rw_len[eid],
                                          self.r_inputs_rels: rr[eid], self.r_inputs_rels_len: rr_len[eid]})
                if np.argmax(sim) == 0:
                    correct += 1
            print '.'
            print correct, '/', num_example, float(correct) / num_example

            saver.save(sess, 'HR_BiLSTM/model-%d.ckpt' % (prev_epoch + epoch))
            self.sess = sess

    def predict(self, q, r):
        '''
        :param q: question, string
        :param r: relation, string
        :return: [0,1]
        '''
        if not self.sess:
            print('Load or train a model first.')
            return -1

        q = [self.word2idx[w] for w in q.split()]
        q_len = np.asarray([len(q)])
        q = np.asarray(q + [0 for i in range(MAX_QUESTION_LENGTH - q_len[0])])
        r_word = [self.word2idx[w] for w in r.replace('_', '.').split('.')]
        r_word_len = np.asarray([len(r_word)])
        r_word = np.asarray(r_word + [0 for i in range(MAX_RELATION_WORD_LEGNTH - r_word_len[0])])
        r_rel = [self.rel2idx[r]]
        r_rel_len = np.asarray([len(r_rel)])
        r_rel = np.asarray(r_rel + [0 for i in range(MAX_RELATION_TYPE_LENGTH - r_rel_len[0])])

        sim = self.sess.run([self.similarity],
                       feed_dict={self.q_inputs: q, self.q_length: q_len,
                                  self.r_inputs_word: r_word, self.r_inputs_word_len: r_word_len,
                                  self.r_inputs_rels: r_rel, self.r_inputs_rels_len: r_rel_len})

        return sim[0]

    def load_model(self, path):
        if self.sess:
            self.sess.close()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.sess, path)


if __name__ == '__main__':
    model = HRBiLSTM()
    model.train(6)


