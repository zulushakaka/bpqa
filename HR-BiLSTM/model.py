import tensorflow as tf
import numpy as np
from data import load_word_embedding


LSTM_HIDDEN_SIZE = 100
MAX_QUESTION_LENGTH = 30
MAX_RELATION_WORD_LEGNTH = 10
MAX_RELATION_TYPE_LENGTH = 5


class HRBiLSTM (object):
    def __init__(self):
        word2idx, embedding_matrix = load_word_embedding()
        self.build_model(embedding_matrix)

    def build_model(self, embedding_matrix):
        # question representation

        q_inputs = tf.placeholder(dtype=tf.int32, shape=[None, MAX_QUESTION_LENGTH])
        # dimension: (batch_size, max_seq_len)
        q_length = tf.placeholder(dtype=tf.int32, shape=[None])
        # dimension: (batcj_size)

        embedding = tf.get_variable(name="embedding", shape=embedding_matrix.shape,
                                         initializer=tf.constant_initializer(embedding_matrix), trainable=False)
        # dimension: (voc_size, embedding_dim)
        embedded = tf.nn.embedding_lookup(embedding, q_inputs)
        # dimension: (batch_size, max_seq_len, embedding_dim)

        with tf.variable_scope('fw1'):
            cell_q1_fw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
        with tf.variable_scope('bw1'):
            cell_q1_bw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
        q1_outputs, _ = tf.nn.bidirectional_dynamic_rnn\
            (cell_fw=cell_q1_fw, cell_bw=cell_q1_bw, inputs=embedded, sequence_length=q_length, dtype=tf.float32)
        q1_outputs = tf.concat(q1_outputs, 2)
        # dimension: (batch_size, max_seq_len, cell_out_size * 2)

        with tf.variable_scope('fw2'):
            cell_q2_fw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
        with tf.variable_scope('bw2'):
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
        r_inputs_rels = tf.placeholder(tf.int32, shape=[None, MAX_RELATION_TYPE_LENGTH])

        r_embedding = load_word_embedding()
        r_embedding = tf.get_variable(name="r_embedding", shape=r_embedding.shape,
                                         initializer=tf.random_normal(r_embedding), trainable=True)
        # relation embeddings are randomly initialized
        r_word_embedded = tf.nn.embedding_lookup(embedding, r_inputs_word)
        r_rels_embedded = tf.nn.embedding_lookup(r_embedding, r_inputs_rels)

        cell_r_fw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
        cell_r_bw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)

        r_word_outputs, r_word_states = tf.nn.bidirectional_dynamic_rnn(cell_r_fw, cell_r_bw, r_word_embedded)
        r_rels_outputs, r_rels_states = tf.nn.bidirectional_dynamic_rnn(cell_r_fw, cell_r_bw, r_rels_embedded)

        r_outputs = tf.concat(tf.concat(r_word_outputs, 2), tf.concat(r_rels_outputs, 2))
        r_pooling = tf.nn.max_pool(r_outputs)

        score = tf.losses.cosine_distance(q_max_pool, r_pooling)

    def train(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    model = HRBiLSTM()