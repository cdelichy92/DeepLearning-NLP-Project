import getpass
import sys
import time
import os

import numpy as np
import tensorflow as tf
#from tensorflow.models.rnn import rnn

from utils import get_snli_dataset, Vocab, get_snli_sentences
from utils import data_iterator_lstm

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64
  embed_size = 100
  hidden_size = 100
  hidden_size2 = 100
  max_epochs = 50
  early_stopping = 2
  dropout = 0.9
  lr = 0.001
  lr_decay = 0.9
  l2 = 0.0005
  label_size = 3
  # sentence length
  # 25 is longer than 95% of the sentences in SNLI, see SPINN paper
  sent_len = 30
  # max norm of the gradient for gradient clipping
  max_grad_norm = 5

  num_layers = 1

class Model():

  def load_vocab(self,debug=False):
    """Loads vocab."""
    self.vocab = Vocab()
    if debug:
        self.vocab.construct(get_snli_dataset('dev'))
    else:
        self.vocab.construct(get_snli_dataset('train'))

  def load_wv(self, filename="glove.6B.100d.txt"):
    """Loads GloVe word-vectors."""
    vocab = self.vocab
    self.wv = np.zeros((len(vocab),self.config.embed_size))
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            word, vector = line.split(' ',1)
            if word in vocab.word_to_index:
                self.wv[vocab.word_to_index[word]] = np.fromstring(vector, sep=' ')
    self.wv[vocab.word_to_index[vocab.unknown]] = -10*np.ones((self.config.embed_size,))
    
    self.wv = self.wv.astype(np.float32)
    
#    print 'test load_wv:'
#    print vocab.word_to_index['was']
#    print vocab.index_to_word[vocab.word_to_index['was']]
#    print self.wv[vocab.word_to_index['was']]
#    print vocab.word_to_index[vocab.padding]
#    print vocab.index_to_word[vocab.word_to_index[vocab.padding]]
#    print self.wv[vocab.word_to_index[vocab.padding]]

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    self.load_vocab(debug=debug)
    self.load_wv(filename="glove.6B.100d.txt")
    config = self.config

    tagnames = ['entailment', 'neutral', 'contradiction']
    self.index_to_tag = dict(enumerate(tagnames))
    tag_to_index = {v:k for k,v in self.index_to_tag.iteritems()}
    
    if debug:
        # Load the training set
        train_data = list(get_snli_sentences(self.vocab, config.sent_len, 'dev'))
        self.sent1_train, self.sent2_train, labels = zip(*train_data)
        self.sent1_train, self.sent2_train = np.vstack(self.sent1_train), np.vstack(self.sent2_train)
        self.y_train = np.array([tag_to_index[label] 
            for label in labels],dtype=np.int32)
        print '# training examples: %d' %len(self.y_train)
        
        # Load the validation set
        dev_data = list(get_snli_sentences(self.vocab, config.sent_len, 'test'))
        self.sent1_dev, self.sent2_dev, labels = zip(*dev_data)
        self.sent1_dev, self.sent2_dev = np.vstack(self.sent1_dev), np.vstack(self.sent2_dev)
        self.y_dev = np.array([tag_to_index[label] 
            for label in labels],dtype=np.int32)
        print '# dev examples: %d' %len(self.y_dev)
        
        # Load the test set
        test_data = list(get_snli_sentences(self.vocab, config.sent_len, 'test'))
        self.sent1_test, self.sent2_test, labels = zip(*test_data)
        self.sent1_test, self.sent2_test = np.vstack(self.sent1_test), np.vstack(self.sent2_test)
        self.y_test = np.array([tag_to_index[label] 
            for label in labels],dtype=np.int32)
        print '# test examples: %d' %len(self.y_test)
    else:
        # Load the training set
        train_data = list(get_snli_sentences(self.vocab, config.sent_len, 'train'))
        self.sent1_train, self.sent2_train, labels_train = zip(*train_data)
        self.sent1_train, self.sent2_train = np.vstack(self.sent1_train), np.vstack(self.sent2_train)
        self.y_train = np.array([tag_to_index[label] 
            for label in labels_train],dtype=np.int32)
        print '# training examples: %d' %len(self.y_train)
        
        # Load the validation set
        dev_data = list(get_snli_sentences(self.vocab, config.sent_len, 'dev'))
        self.sent1_dev, self.sent2_dev, labels_dev = zip(*dev_data)
        self.sent1_dev, self.sent2_dev = np.vstack(self.sent1_dev), np.vstack(self.sent2_dev)
        self.y_dev = np.array([tag_to_index[label] 
            for label in labels_dev],dtype=np.int32)
        print '# dev examples: %d' %len(self.y_dev)
        n_entail = len(self.y_dev[self.y_dev==0])
        n_neut = len(self.y_dev[self.y_dev==1])
        n_contr = len(self.y_dev[self.y_dev==2])
        print 'entail: %d' %n_entail
        print 'neut: %d' %n_neut
        print 'contr: %d' %n_contr

#        for idx in range(0,5):
#            sent_idx = list(self.sent1_dev[idx,:])
#            print ' '.join([self.vocab.index_to_word[idw] for idw in sent_idx])
#            print self.index_to_tag[self.y_dev[idx]]
        
        # Load the test set
        test_data = list(get_snli_sentences(self.vocab, config.sent_len, 'test'))
        self.sent1_test, self.sent2_test, labels_test = zip(*test_data)
        self.sent1_test, self.sent2_test = np.vstack(self.sent1_test), np.vstack(self.sent2_test)
        self.y_test = np.array([tag_to_index[label] 
            for label in labels_test],dtype=np.int32)
        print '# test examples: %d' %len(self.y_test)
           
  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors
    """
    
    self.sent1_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.sent_len],
                                            name='sent1')
    self.sent2_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.sent_len],
                                            name='sent2')
    self.labels_placeholder = tf.placeholder(tf.float32,
                                             shape=[None, self.config.label_size], name='label')
    self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')
    
  def create_feed_dict(self, sent1_batch, sent2_batch, dropout, label_batch):
    """Creates the feed_dict for the model
    """

    feed_dict = {
        self.sent1_placeholder: sent1_batch,
        self.sent2_placeholder: sent2_batch,
        self.labels_placeholder: label_batch,
        self.dropout_placeholder: dropout
    }

    return feed_dict
    
  def add_embedding(self):
    """Add embedding layer.

    Returns:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    """    
    
    self.embeddings = tf.convert_to_tensor(self.wv)    
    
    with tf.device('/cpu:0'):
        embedding = tf.get_variable('Embedding', 
                                    trainable=False, initializer=self.embeddings)
        sent1_inputs = tf.nn.embedding_lookup(embedding, self.sent1_placeholder)
        sent2_inputs = tf.nn.embedding_lookup(embedding, self.sent2_placeholder)
    
    print 'sentence inputs after embedding lookup:'
    print sent1_inputs.get_shape()
    # shape : (batch_size, sent_len, embed_size)
            
    return sent1_inputs, sent2_inputs
    
  def compute_sentence_repr_avg(self, sent1, sent2):
    
    output1 = tf.reduce_sum(sent1, 1)
    print 'sentence repr after avging:'
    print output1.get_shape()
        
    output2 = tf.reduce_sum(sent2, 1)

    return output1, output2
    
  def attention_model(self, sent1, sent2):
    
    config = self.config
    hidden_size = config.hidden_size
    batch_size = config.batch_size
    sent_len = config.sent_len
    
    with tf.variable_scope('LSTM1'):
        lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)      
        lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell1, output_keep_prob=self.dropout_placeholder)
        cell1 = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1] * config.num_layers)
    
        self.initial_state = cell1.zero_state(batch_size, tf.float32)
                
    outputs1 = []
    state = self.initial_state
    with tf.variable_scope("RNN1"):
      for time_step in range(config.sent_len):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell1(sent1[:, time_step, :], state)
        outputs1.append(cell_output)
    
    final_state1 = state
        
    with tf.variable_scope('LSTM2'):
        lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
        lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell2, output_keep_prob=self.dropout_placeholder)
        cell2 = tf.nn.rnn_cell.MultiRNNCell([lstm_cell2] * config.num_layers)
    
    outputs2 = []
    state = final_state1
    with tf.variable_scope("RNN2"):
      for time_step in range(config.sent_len):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell2(sent2[:, time_step, :], state)
        outputs2.append(cell_output)
    
    Y = tf.pack(outputs1)
    print "shape1:"
    print Y.get_shape()
    Y = tf.transpose(Y, perm=[1,2,0])
    print "shape2:"
    print Y.get_shape()
    
    hN = outputs2[-1]
    
    H = tf.pack(list(tf.split(1, config.sent_len, tf.tile(hN, [1,config.sent_len]))))
    print "shape1:"
    print H.get_shape()
    H = tf.transpose(H, perm=[1,2,0])
    print "shape2:"
    print H.get_shape()
    
    with tf.variable_scope('AttentionLayer'):
      Wy = tf.get_variable('Wy', [config.hidden_size, config.hidden_size])
      Wh = tf.get_variable('Wh', [config.hidden_size, config.hidden_size])
      Wy_extended = tf.pack(list(tf.split(1, config.batch_size, tf.tile(Wy, [1,config.batch_size]))))
      Wh_extended = tf.pack(list(tf.split(1, config.batch_size, tf.tile(Wh, [1,config.batch_size]))))
      M = tf.nn.tanh(tf.batch_matmul(Wy_extended, Y) + 
                     tf.batch_matmul(Wh_extended, H))
      w = tf.get_variable('w', [config.hidden_size])
#      w_extended = tf.pack(list(tf.split(0, config.batch_size, tf.tile(w, [config.batch_size]))))
#      print w_extended
      print M
      temp = tf.matmul(tf.reshape(w, [1,-1]), tf.reshape(M, [hidden_size,-1]))
      temp = tf.reshape(temp, [config.batch_size, config.sent_len])
      alpha = tf.nn.softmax(temp)
      #alpha = tf.nn.softmax(tf.batch_matmul(w_extended, M))
      print alpha
      #temp = tf.matmul(tf.reshape(Y, [config.batch_size * config.hidden_size, config.sent_len]), 
      #                 tf.transpose(alpha))
      r = tf.squeeze(tf.batch_matmul(Y, tf.reshape(alpha, [batch_size, -1, 1])))
      #r = tf.reshape(temp, [config.batch_size, config.hidden_size])
      print r
      
      #r = tf.batch_matmul(Y, alpha)
      
      Wp = tf.get_variable('Wp', [config.hidden_size, config.hidden_size])
      Wx = tf.get_variable('Wx', [config.hidden_size, config.hidden_size])
      hs = tf.nn.tanh(tf.matmul(r, Wp) + tf.matmul(hN, Wx))
    
    return hs


  def compute_sentence_repr_lstm(self, sent1, sent2):
      
    config = self.config
    hidden_size = config.hidden_size
    batch_size = config.batch_size

    with tf.variable_scope('LSTM'):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)      
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=self.dropout_placeholder)
        
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
    
        self.initial_state = cell.zero_state(batch_size, tf.float32)
    
        #sent1 = tf.nn.dropout(sent1, self.dropout_placeholder)
#        sentence = [tf.squeeze(sent_, [1]) 
#            for sent_ in tf.split(1, self.config.sent_len, sentence)]
                
    outputs1 = []
    state = self.initial_state
    with tf.variable_scope("RNN1"):
      for time_step in range(config.sent_len):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(sent1[:, time_step, :], state)
        outputs1.append(cell_output)

        #outputs, state = rnn.rnn(cell, sentence, initial_state=self.initial_state)
        
    #output1 = tf.add_n(outputs1)
    
    outputs2 = []
    state = self.initial_state
    with tf.variable_scope("RNN2"):
      for time_step in range(config.sent_len):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(sent2[:, time_step, :], state)
        outputs2.append(cell_output)
    
    #output2 = tf.add_n(outputs2)

    return outputs1[-1], outputs2[-1]

  def add_model(self, sent1, sent2):
    """Adds the 1-hidden-layer NN.

    Args:
      x: tf.Tensor of shape (-1, 2*embed_size)
    Returns:
      output: tf.Tensor of shape (batch_size, label_size)
    """
    
    config = self.config    
    
    #sentv1, sentv2 = self.compute_sentence_repr_lstm(sent1, sent2)
    #sentv1, sentv2 = self.compute_sentence_repr_avg(sent1, sent2)
    
#    print 'sentv1:'
#    print sentv1.get_shape()
#    
#    x = tf.concat(1, [sentv1, sentv2])
#    print 'x:'
#    print x.get_shape()
    
    x = self.attention_model(sent1, sent2)

    with tf.variable_scope('Layer1'):
        # For the classic LSTM
#      W1 = tf.get_variable('W1', [2 * config.hidden_size, config.hidden_size2], 
#                          initializer=tf.contrib.layers.xavier_initializer() )
      W1 = tf.get_variable('W1', [config.hidden_size, config.hidden_size2], 
                           initializer=tf.contrib.layers.xavier_initializer() )
      b1 = tf.get_variable('b1', [config.hidden_size2])
      h1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
      h1 = tf.nn.dropout(h1, self.dropout_placeholder)
#      if config.l2:
#          tf.add_to_collection('total_loss', 0.5 * config.l2 * tf.nn.l2_loss(W))
    
    with tf.variable_scope('Layer2'):
      W2 = tf.get_variable('W2', [config.hidden_size2, config.hidden_size2], 
                          initializer=tf.contrib.layers.xavier_initializer() )
      b2 = tf.get_variable('b2', [config.hidden_size2])
      h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
      h2 = tf.nn.dropout(h2, self.dropout_placeholder)

    with tf.variable_scope('ScoreLayer'):
      U = tf.get_variable('U', [config.hidden_size2, config.label_size],
                          initializer=tf.contrib.layers.xavier_initializer() )
      b2 = tf.get_variable('b2', [config.label_size])
      logits = tf.matmul(h2, U) + b2
#      if config.l2:
#          tf.add_to_collection('total_loss', 0.5 * config.l2 * tf.nn.l2_loss(U))
    
    return logits

  def add_loss_op(self, logits):
    """Adds cross_entropy_loss ops to the computational graph.

    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, self.labels_placeholder))
#    tf.add_to_collection('total_loss', cross_entropy)
#    loss = tf.add_n(tf.get_collection('total_loss'))
    loss = cross_entropy
    
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      self.config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars)) 
#    optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
#    train_op = optimizer.minimize(loss)   

    return train_op
  
  def __init__(self, config):
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    self.sent1_inputs, self.sent2_inputs = self.add_embedding()

    self.logits = self.add_model(self.sent1_inputs, self.sent2_inputs)

    self.loss = self.add_loss_op(self.logits)
    self.predictions = tf.nn.softmax(self.logits)
    one_hot_prediction = tf.argmax(self.predictions, 1)
    correct_prediction = tf.equal(
        tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
    self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
    self.train_op = self.add_training_op(self.loss)
    
  def run_epoch(self, session, sent1_data, sent2_data, input_labels,
                shuffle=True, verbose=100):
    orig_sent1, orig_sent2, orig_y = sent1_data, sent2_data, input_labels
    dp = self.config.dropout
    
    total_loss = []
    total_correct_examples = 0
    total_processed_examples = 0
    total_steps = int( orig_sent1.shape[0] / self.config.batch_size)
    for step, (sent1, sent2, y) in enumerate(
      data_iterator_lstm( orig_sent1, orig_sent2, orig_y, batch_size=self.config.batch_size,
                   label_size=self.config.label_size, shuffle=shuffle)):
      feed = self.create_feed_dict(sent1, sent2, dp, y)
      loss, total_correct, _ = session.run(
          [self.loss, self.correct_predictions, self.train_op],
          feed_dict=feed)
      total_processed_examples += len(y)
      total_correct_examples += total_correct
      total_loss.append(loss)
      ##
      if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : loss = {}'.format(
            step, total_steps, np.mean(total_loss)))
        sys.stdout.flush()
    if verbose:
        sys.stdout.write('\r')
        sys.stdout.flush()
    return np.mean(total_loss), total_correct_examples / float(total_processed_examples), total_loss

  def predict(self, session, sent1_data, sent2_data, y=None):
    """Make predictions from the provided model."""
    # If y is given, the loss is also calculated
    # We deactivate dropout by setting it to 1
    dp = 1
    losses = []
    results = []
    if np.any(y):
        data = data_iterator_lstm(sent1_data, sent2_data, y, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    else:
        data = data_iterator_lstm(sent1_data, sent2_data, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    for step, (sent1, sent2, y) in enumerate(data):
      feed = self.create_feed_dict(sent1, sent2, dp, y)
      if np.any(y):
        loss, preds = session.run(
            [self.loss, self.predictions], feed_dict=feed)
        losses.append(loss)
      else:
        preds = session.run(self.predictions, feed_dict=feed)
      predicted_indices = preds.argmax(axis=1)
      results.extend(predicted_indices)
    return np.mean(losses), results

def print_confusion(confusion, index_to_tag):
    """Helper method that prints confusion matrix."""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print
    print confusion
    val_acc = 0.0
    for i, tag in sorted(index_to_tag.items()):
        val_acc += confusion[i, i]
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print 'Tag: {} - P {:2.4f} / R {:2.4f}'.format(tag, prec, recall)
    val_acc /= confusion.sum()
    return val_acc

def calculate_confusion(config, predicted_indices, y_indices):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((config.label_size, config.label_size), dtype=np.int32)
    for i in xrange(len(y_indices)):
        try:
            correct_label = y_indices[i]
            guessed_label = predicted_indices[i]
            confusion[correct_label, guessed_label] += 1
        except:
            continue
    return confusion
    
def test_Model():
  """Test model implementation.
  """
  complete_loss_history = []
  train_acc_history = []
  val_acc_history = []
  config = Config()
  with tf.Graph().as_default():
    model = Model(config)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
      best_val_loss = float('inf')
      best_val_epoch = 0

      session.run(init)
      for epoch in xrange(config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()
        ###
        train_loss, train_acc, loss_history = model.run_epoch(session, 
                                                model.sent1_train,
                                                model.sent2_train, 
                                                model.y_train)
        val_loss, predictions = model.predict(session,
                                              model.sent1_dev,
                                              model.sent2_dev, 
                                              model.y_dev)
        complete_loss_history.extend(loss_history)
        train_acc_history.append(train_acc)
        print 'Training loss: {}'.format(train_loss)
        print 'Training acc: {}'.format(train_acc)
        print 'Validation loss: {}'.format(val_loss)
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_val_epoch = epoch
          if not os.path.exists("./weights"):
            os.makedirs("./weights")

          saver.save(session, './weights/lstm.weights')
        if epoch - best_val_epoch > config.early_stopping:
          break
        ##
        confusion = calculate_confusion(config, predictions, model.y_dev)
        val_acc = print_confusion(confusion, model.index_to_tag)
        val_acc_history.append(val_acc)
        print 'Validation acc: {}'.format(val_acc)
        print 'Total time: {}'.format(time.time() - start)
        config.lr = config.lr_decay * config.lr
      
      saver.restore(session, './weights/lstm.weights')
  with open('complete_loss_history.txt', 'w') as clh:
      for loss in complete_loss_history:
          clh.write("%f " % loss)
      
def little_tests():
    config = Config()
    model = Model(config)

if __name__ == "__main__":
    #little_tests()
    test_Model()
    
