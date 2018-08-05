import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
import pickle

import numpy as np
import tensorflow as tf

from utils import get_words_dataset, Vocab, get_sentences_dataset
from utils import data_iterator, INDEX_TO_LABEL
import utils

class Config(object):
    """
        Holds model hyperparams and data information.
        The config class is used to store various hyperparameters and dataset
        information parameters.
        Model objects are passed a Config() object at
        instantiation.
    """
    batch_size = 64
    word_embed_size = 100
    sentence_embed_size = 100
    hidden_sizes = [128, 32]
    max_epochs = 50
    early_stopping = 3
    kp = 0.9
    lr = 0.0002
    l2 = 0.000
    label_size = 3

    # sentence length
    # 25 is longer than 95% of the sentences in SNLI, see SPINN paper
    sent_len = 25

    # max norm of the gradient for gradient clipping
    max_grad_norm = 5
    num_layers = 1


class Model():


    def __init__(self, config):
        self.config = config
        self.load_data()
        self.build_model()


    def load_vocab(self,debug):
        self.vocab = Vocab()
        if debug:
            self.vocab.construct(get_words_dataset('dev'))
        else:
            self.vocab.construct(get_words_dataset('train'))
        self.vocab.build_embedding_matrix(self.config.word_embed_size)
        self.embedding_matrix = self.vocab.embedding_matrix


    def load_data(self, debug=False):
        """
            Loads starter word-vectors and train/dev/test data.
        """
        self.load_vocab(debug)
        config = self.config

        if debug:
            # Load the training set
            train_data = list(get_sentences_dataset(self.vocab,
                config.sent_len, 'dev', 'post'))
            ( self.sent1_train, self.sent2_train, self.len1_train,
                self.len2_train, self.y_train ) = zip(*train_data)
            self.sent1_train, self.sent2_train = np.vstack(self.sent1_train), np.vstack(self.sent2_train)
            self.len1_train, self.len2_train = ( np.array(self.len1_train),
                np.array(self.len2_train) )
            self.y_train = np.array(self.y_train)
            print('# training examples: %d' %len(self.y_train))

            # Load the validation set
            dev_data = list(get_sentences_dataset(self.vocab, config.sent_len,
                'test', 'post'))
            ( self.sent1_dev, self.sent2_dev, self.len1_dev,
                self.len2_dev, self.y_dev ) = zip(*dev_data)
            self.sent1_dev, self.sent2_dev = np.vstack(self.sent1_dev), np.vstack(self.sent2_dev)
            self.len1_dev, self.len2_dev = ( np.array(self.len1_dev),
                np.array(self.len2_dev) )
            self.y_dev = np.array(self.y_dev)
            print('# dev examples: %d' %len(self.y_dev))

            # Load the test set
            test_data = list(get_sentences_dataset(self.vocab, config.sent_len,
                'test', 'post'))
            ( self.sent1_test, self.sent2_test, self.len1_test,
                self.len2_test, self.y_test ) = zip(*test_data)
            self.sent1_test, self.sent2_test = np.vstack(self.sent1_test), np.vstack(self.sent2_test)
            self.len1_test, self.len2_test = ( np.array(self.len1_test),
                np.array(self.len2_test) )
            self.y_test = np.array(self.y_test)
            print('# test examples: %d' %len(self.y_test))
        else:
            # Load the training set
            train_data = list(get_sentences_dataset(self.vocab,
                config.sent_len, 'train', 'post'))
            ( self.sent1_train, self.sent2_train, self.len1_train,
                self.len2_train, self.y_train ) = zip(*train_data)
            self.sent1_train, self.sent2_train = np.vstack(self.sent1_train), np.vstack(self.sent2_train)
            self.len1_train, self.len2_train = ( np.array(self.len1_train),
                np.array(self.len2_train) )
            self.y_train = np.array(self.y_train)
            print('# training examples: %d' %len(self.y_train))

            # Load the validation set
            dev_data = list(get_sentences_dataset(self.vocab, config.sent_len,
                'dev', 'post'))
            ( self.sent1_dev, self.sent2_dev, self.len1_dev,
                self.len2_dev, self.y_dev ) = zip(*dev_data)
            self.sent1_dev, self.sent2_dev = np.vstack(self.sent1_dev), np.vstack(self.sent2_dev)
            self.len1_dev, self.len2_dev = ( np.array(self.len1_dev),
                np.array(self.len2_dev) )
            self.y_dev = np.array(self.y_dev)
            print('# dev examples: %d' %len(self.y_dev))

            # Load the test set
            test_data = list(get_sentences_dataset(self.vocab, config.sent_len,
                'test', 'post'))
            ( self.sent1_test, self.sent2_test, self.len1_test,
                self.len2_test, self.y_test ) = zip(*test_data)
            self.sent1_test, self.sent2_test = np.vstack(self.sent1_test), np.vstack(self.sent2_test)
            self.len1_test, self.len2_test = ( np.array(self.len1_test),
                np.array(self.len2_test) )
            self.y_test = np.array(self.y_test)
            print('# test examples: %d' %len(self.y_test))

            print('min len: ', np.min(self.len2_train))


    def build_model(self):
        config = self.config
        k = config.sentence_embed_size
        L = config.sent_len

        # input tensors
        self.sent1_ph = tf.placeholder(tf.int32, shape=[None, L],
                                       name='sent1')
        self.sent2_ph = tf.placeholder(tf.int32, shape=[None, L],
                                       name='sent2')
        self.len1_ph = tf.placeholder(tf.int32, shape=[None], name='len1')
        self.len2_ph = tf.placeholder(tf.int32, shape=[None], name='len2')
        self.labels_ph = tf.placeholder(tf.float32,
                                        shape=[None, config.label_size],
                                        name='label')
        self.kp_ph = tf.placeholder(tf.float32, name='kp')
        kp = self.kp_ph

        # set embedding matrix to pretrained embedding
        init_embeds = tf.constant(self.embedding_matrix, dtype='float32')
        word_embeddings = tf.get_variable(
                dtype='float32',
                name='word_embeddings',
                initializer=init_embeds,
                trainable=False) # no fine-tuning of word embeddings

        # x1 and x2 have shape (?, L, k)
        x1 = tf.nn.embedding_lookup(word_embeddings, self.sent1_ph)
        x2 = tf.nn.embedding_lookup(word_embeddings, self.sent2_ph)
        x1, x2 = tf.nn.dropout(x1, kp), tf.nn.dropout(x2, kp)

        # encode premise sentence with 1st LSTM
        with tf.variable_scope('rnn1'):
            cell1 = tf.contrib.rnn.LSTMCell(num_units=k,
                    state_is_tuple=True)
            out1, fstate1 = tf.nn.dynamic_rnn(
                cell=cell1,
                inputs=x1,
                sequence_length=self.len1_ph,
                dtype=tf.float32)

        # encode hypothesis with 2nd LSTM
        # using final state of 1st LSTM as initial state
        with tf.variable_scope('rnn2'):
            cell2 = tf.contrib.rnn.LSTMCell(num_units=k,
                    state_is_tuple=True)
            out2, fstate2 = tf.nn.dynamic_rnn(
                cell=cell2,
                inputs=x2,
                sequence_length=self.len2_ph,
                initial_state=fstate1,
                dtype=tf.float32)

        Y = out1
        Y_mod =tf.reshape(Y, [-1, k])

        W_y = tf.get_variable(name='W_y', shape=[k, k],
                regularizer=tf.contrib.layers.l2_regularizer(config.l2))
        W_h = tf.get_variable(name='W_h', shape=[k, k],
                regularizer=tf.contrib.layers.l2_regularizer(config.l2))
        b_M = tf.get_variable(name='b_M', initializer=tf.zeros([L, k]))
        W_r = tf.get_variable(name='W_r', shape=[k, k],
                regularizer=tf.contrib.layers.l2_regularizer(config.l2))
        W_t = tf.get_variable(name='W_t', shape=[k, k],
                regularizer=tf.contrib.layers.l2_regularizer(config.l2))
        b_r = tf.get_variable(name='b_r', initializer=tf.zeros([k]))
        w = tf.get_variable(name='w', shape=[k, 1],
                regularizer=tf.contrib.layers.l2_regularizer(config.l2))
        b_a = tf.get_variable(name='b_a', initializer=tf.zeros([L]))

        rt_1 = tf.zeros([tf.shape(self.len1_ph)[0], k])
        attention = []
        r_outputs = []
        for t in range(L):
            ht = out2[:,t,:]

            Ht = tf.reshape(tf.tile(ht, [1, L]), [-1, L, k])
            Ht_mod = tf.reshape(Ht, [-1, k])
            Rt_1 = tf.reshape(tf.tile(rt_1, [1, L]), [-1, L, k])
            Rt_1_mod = tf.reshape(Rt_1, [-1, k])
            Mt = tf.nn.tanh( tf.reshape(tf.matmul(Y_mod, W_y),
                                 [-1, L, k]) +
                             tf.reshape(tf.matmul(Ht_mod, W_h),
                                 [-1, L, k]) +
                             tf.reshape(tf.matmul(Rt_1_mod, W_r),
                                 [-1, L, k])  + b_M)
            Mt_w = tf.matmul(tf.reshape(Mt, [-1, k]), w)
            alphat = tf.nn.softmax(tf.reshape(Mt_w, [-1, 1, L]) + b_a)
            alphat_Y = tf.reshape(tf.matmul(alphat, Y), [-1, k])
            rt = alphat_Y + tf.nn.tanh(tf.matmul(rt_1, W_t) + b_r)
            rt_1 = rt
            attention.append(alphat)
            r_outputs.append(rt)

        r_outputs = tf.stack(r_outputs)
        self.attention = tf.stack(attention)
        r_outputs = tf.transpose(r_outputs, [1, 0, 2])

        def get_last_relevant_output(out, seq_len):
            rng = tf.range(0, tf.shape(seq_len)[0])
            indx = tf.stack([rng, seq_len - 1], 1)
            last = tf.gather_nd(out, indx)
            return last

        rN = get_last_relevant_output(r_outputs, self.len2_ph)
        hN = get_last_relevant_output(out2, self.len2_ph)

        W_p = tf.get_variable(name='W_p', shape=[k, k],
                regularizer=tf.contrib.layers.l2_regularizer(config.l2))
        W_x = tf.get_variable(name='W_x', shape=[k, k],
                regularizer=tf.contrib.layers.l2_regularizer(config.l2))
        b_hs = tf.get_variable(name='b_hs', initializer=tf.zeros([k]))

        # sentence pair representation
        h_s = tf.nn.tanh(tf.matmul(rN, W_p) + tf.matmul(hN, W_x) + b_hs)

        y = h_s

        # MLP classifier on top
        hidden_sizes = config.hidden_sizes
        for layer, size in enumerate(hidden_sizes):
            if layer > 0:
                previous_size = hidden_sizes[layer-1]
            else:
                previous_size = k
            W = tf.get_variable(name='W{}'.format(layer),
                    shape=[previous_size, size],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(config.l2))
            b = tf.get_variable(name='b{}'.format(layer),
                    initializer=tf.zeros([size]))
            y = tf.nn.relu(tf.matmul(y, W) + b)

        W_softmax = tf.get_variable(name='W_softmax',
                shape=[hidden_sizes[-1], config.label_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(config.l2))
        b_softmax = tf.get_variable(name='b_softmax',
                initializer=tf.zeros([config.label_size]))

        logits = tf.matmul(y, W_softmax) + b_softmax
        cross_entropy_loss = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(self.labels_ph, logits)
                )
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = cross_entropy_loss #+ tf.add_n(reg_losses)

        self.train_op = ( tf.train.AdamOptimizer(learning_rate=config.lr)
                .minimize(self.loss) )

        self.probs = tf.nn.softmax(logits)
        self.predictions = tf.argmax(self.probs, 1)
        correct_prediction = tf.equal(
            tf.argmax(self.labels_ph, 1), self.predictions)
        self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))


    def create_feed_dict(self, sent1_batch, sent2_batch, len1_batch,
            len2_batch, label_batch, keep_prob):
        feed_dict = {
            self.sent1_ph: sent1_batch,
            self.sent2_ph: sent2_batch,
            self.len1_ph: len1_batch,
            self.len2_ph: len2_batch,
            self.labels_ph: label_batch,
            self.kp_ph: keep_prob
        }
        return feed_dict


    def run_epoch(self, session, sent1_data, sent2_data, len1_data, len2_data, input_labels,
            verbose=100):
        orig_sent1, orig_sent2, orig_len1, orig_len2, orig_y = ( sent1_data,
                sent2_data, len1_data, len2_data, input_labels )
        kp = self.config.kp
        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = int( orig_sent1.shape[0] / self.config.batch_size)
        for step, (sent1, sent2, len1, len2, y) in enumerate(
            data_iterator(orig_sent1, orig_sent2, orig_len1, orig_len2, orig_y,
                    batch_size=self.config.batch_size, label_size=self.config.label_size)):
            feed = self.create_feed_dict(sent1, sent2, len1, len2, y, kp)
            loss, total_correct, _ = session.run(
                [self.loss, self.correct_predictions, self.train_op],
                feed_dict=feed)
            total_processed_examples += len(y)
            total_correct_examples += total_correct
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()
        return np.mean(total_loss), total_correct_examples / float(total_processed_examples), total_loss


    def predict(self, session, sent1_data, sent2_data, len1_data, len2_data, y=None):
        """Make predictions from the provided model."""
        # If y is given, the loss is also calculated
        # We deactivate dropout by setting it to 1
        kp = 1.0
        losses = []
        results = []
        if np.any(y):
            data = data_iterator(sent1_data, sent2_data, len1_data, len2_data, y, batch_size=self.config.batch_size,
                                 label_size=self.config.label_size, shuffle=False)
        else:
            data = data_iterator(sent1_data, sent2_data, len1_data, len2_data, batch_size=self.config.batch_size,
                                 label_size=self.config.label_size, shuffle=False)
        for step, (sent1, sent2, len1, len2, y) in enumerate(data):
            feed = self.create_feed_dict(sent1, sent2, len1, len2, y, kp)
            if np.any(y):
                loss, preds = session.run(
                    [self.loss, self.predictions], feed_dict=feed)
                losses.append(loss)
            else:
                preds = session.run(self.predictions, feed_dict=feed)
            results.extend(preds)
        return np.mean(losses), np.array(results)


    def get_attention(self, session, sent1, sent2):
        kp = 1.0
        sent1 = utils.encode_sentence(self.vocab, sent1)
        print(sent1)
        sent2 = utils.encode_sentence(self.vocab, sent2)
        print(sent2)
        sent1 = utils.pad_sentence(self.vocab, sent1, self.config.sent_len,
                'post')
        sent2 = utils.pad_sentence(self.vocab, sent2, self.config.sent_len,
                'post')
        len1, len2 = np.array([len(sent1)]), np.array([len(sent2)])
        sent1_arr = np.array(sent1).reshape((1,-1))
        sent2_arr = np.array(sent2).reshape((1,-1))
        y = np.array([0,1,0]).reshape((1,-1))
        feed = self.create_feed_dict(sent1_arr, sent2_arr, len1, len2, y, kp)
        preds, alphas = session.run([self.predictions, self.attention], feed_dict=feed)
        return preds, alphas


def print_confusion(confusion):
    """Helper method that prints confusion matrix."""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print(confusion)
    val_acc = 0.0
    for i in range(3):
        tag = INDEX_TO_LABEL[i]
        val_acc += confusion[i, i]
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print('Tag: {} - P {:2.4f} / R {:2.4f}'.format(tag, prec, recall))
    val_acc /= confusion.sum()
    return val_acc


def calculate_confusion(config, predicted_indices, y_indices):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((config.label_size, config.label_size), dtype=np.int32)
    for i in range(len(y_indices)):
        try:
            correct_label = y_indices[i]
            guessed_label = predicted_indices[i]
            confusion[correct_label, guessed_label] += 1
        except:
            continue
    return confusion


def train_model():
    """
        Trains the model
    """
    complete_loss_history = []
    train_acc_history = []
    val_acc_history = []
    config = Config()
    with tf.Graph().as_default():
        model = Model(config)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            best_val_loss = float('inf')
            best_val_epoch = 0

            session.run(init)
            #saver.restore(session, './weights/lstm.weights')
            for epoch in range(config.max_epochs):
                print('Epoch {}'.format(epoch))
                start = time.time()
                train_loss, train_acc, loss_history = model.run_epoch(session,
                                                            model.sent1_train,
                                                            model.sent2_train,
                                                            model.len1_train,
                                                            model.len2_train,
                                                            model.y_train)
                val_loss, predictions = model.predict(session,
                                                      model.sent1_dev,
                                                      model.sent2_dev,
                                                      model.len1_dev,
                                                      model.len2_dev,
                                                      model.y_dev)
                val_acc = np.mean(np.equal(predictions, model.y_dev))
                complete_loss_history.extend(loss_history)
                train_acc_history.append(train_acc)
                print('Training loss: {}'.format(train_loss))
                print('Training acc: {}'.format(train_acc))
                print('Validation loss: {}'.format(val_loss))
                print('Validation acc: {}'.format(val_acc))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    if not os.path.exists("./weights"):
                        os.makedirs("./weights")
                    saver.save(session, './weights/lstm.weights')
                if epoch - best_val_epoch > config.early_stopping:
                    break
    confusion = calculate_confusion(config, predictions, model.y_dev)
    val_acc = print_confusion(confusion)
    val_acc_history.append(val_acc)
    print('Validation acc: {}'.format(val_acc))
    print('Total time: {}'.format(time.time() - start))
    with open('complete_loss_history.txt', 'w') as clh:
        for loss in complete_loss_history:
            clh.write("%f " % loss)


def test_attention():
    sentences1 = []
    sentences2 = []
    predictions = []
    attentions = []
    config = Config()
    with tf.Graph().as_default():
        model = Model(config)
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, './weights/lstm.weights')
            for line in open('sentences.txt', 'r'):
                sent1, sent2 = line.split('|')
                sent1, sent2 = sent1.strip(), sent2.strip()
                preds, attention = model.get_attention(session, sent1, sent2)
                attention = np.squeeze(attention)
                print(sent1)
                print(sent2)
                print(attention)
                sentences1.append(sent1)
                sentences2.append(sent2)
                predictions.append(preds)
                attentions.append(attention)
    pickle.dump((sentences1, sentences2, predictions, attentions),
            open('attention_results.pkl', 'wb'))


if __name__ == "__main__":
    train_model()
    test_attention()
