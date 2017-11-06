import tensorflow as tf
import numpy as np
from data import load_word_embedding


LSTM_HIDDEN_SIZE = 100


class HRBiLSTM (object):
    def __init__(self):
        q_inputs = tf.placeholder(tf.int32, shape=(None))  # inputs are 1-dim integers

        embedding = load_word_embedding()
        embedding = tf.get_variable(name="embedding", shape=embedding.shape,
                                         initializer=tf.constant_initializer(embedding), trainable=False)
        embedded = tf.nn.embedding_lookup(embedding, q_inputs)

        cell_q1_fw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
        cell_q1_bw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
        q1_outputs, q1_states = tf.nn.bidirectional_dynamic_rnn(cell_q1_fw, cell_q1_bw, embedded)

        cell_q2_fw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
        cell_q2_bw = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
        q2_outputs, q2_states = tf.nn.bidirectional_dynamic_rnn(cell_q2_fw, cell_q2_bw, tf.concat(q1_outputs, 2))

        q_outputs = tf.add(q1_outputs, q2_outputs)
        q_pooling = tf.nn.max_pool(q_outputs)

        r_inputs_word = tf.placeholder(tf.int32, shape=(None))
        r_inputs_rels = tf.placeholder(tf.int32, shape=(None))

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

        score = tf.losses.cosine_distance(q_pooling, r_pooling)

    def train(self):
        pass

    def predict(self):
        pass
