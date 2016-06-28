import getpass
import sys
import time
import os

import numpy as np
import tensorflow as tf

from utils import get_snli_dataset, Vocab, get_snli_sentences, get_features
from utils import data_iterator

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64
  embed_size = 100
  hidden_size = 60
  max_epochs = 16
  early_stopping = 2
  dropout = 0.9
  lr = 0.001
  #previously: lr=0.001 (train acc 61.26% at epch 15)
  l2 = 0.0005
  label_size = 3

class Model():

  def load_vocab(self):
    """Loads vocab."""
    self.vocab = Vocab()
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
    
#    print 'test load_wv:'
#    print vocab.word_to_index['was']
#    print vocab.index_to_word[vocab.word_to_index['was']]
#    print self.wv[vocab.word_to_index['was']]

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    self.load_vocab()
    self.load_wv(filename="glove.6B.100d.txt")
    vocab = self.vocab
    wv = self.wv

    tagnames = ['entailment', 'neutral', 'contradiction']
    self.index_to_tag = dict(enumerate(tagnames))
    tag_to_index = {v:k for k,v in self.index_to_tag.iteritems()}

    # Load the training set
    self.X_train = np.array([get_features(vocab.word_to_index, wv, sent1, sent2) 
        for sent1,sent2,_ in get_snli_sentences('train')])
    self.y_train = np.array([tag_to_index[label] 
        for _,_,label in get_snli_sentences('train')],dtype=np.int32)
    print '# training examples: %d' %len(self.y_train)
            
    # Load the dev set
    self.X_dev = np.array([get_features(vocab.word_to_index, wv, sent1, sent2) 
        for sent1,sent2,_ in get_snli_sentences('dev')])
    self.y_dev = np.array([tag_to_index[label] 
        for _,_,label in get_snli_sentences('dev')],dtype=np.int32)
    print '# dev examples: %d' %len(self.y_dev)
    
    # Load the test set
    self.X_test = np.array([get_features(vocab.word_to_index, wv, sent1, sent2) 
        for sent1,sent2,_ in get_snli_sentences('test')])
    self.y_test = np.array([tag_to_index[label] 
        for _,_,label in get_snli_sentences('test')],dtype=np.int32)
    print '# test examples: %d' %len(self.y_test)
            
  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph

    input_placeholder: Input placeholder tensor of shape
                       (None, window_size), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, label_size), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    self.input_placeholder = tf.placeholder(
        tf.float32, shape=[None, 2*self.config.embed_size], name='Input')
    self.labels_placeholder = tf.placeholder(
        tf.float32, shape=[None, self.config.label_size], name='Target')
    self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
    
  def create_feed_dict(self, input_batch, dropout, label_batch=None):
    """Creates the feed_dict for softmax classifier.

    A feed_dict takes the form of:

    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }


    Hint: The keys for the feed_dict should be a subset of the placeholder
          tensors created in add_placeholders.
    Hint: When label_batch is None, don't add a labels entry to the feed_dict.
    
    Args:
      input_batch: A batch of input data.
      label_batch: A batch of label data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """

    feed_dict = {
        self.input_placeholder: input_batch,
    }
    if label_batch is not None:
      feed_dict[self.labels_placeholder] = label_batch
    if dropout is not None:
      feed_dict[self.dropout_placeholder] = dropout

    return feed_dict

  def add_model(self, x):
    """Adds the 1-hidden-layer NN.

    Hint: Use a variable_scope (e.g. "Layer") for the first hidden layer, and
          another variable_scope (e.g. "Softmax") for the linear transformation
          preceding the softmax. Make sure to use the xavier_weight_init you
          defined in the previous part to initialize weights.
    Hint: Make sure to add in regularization and dropout to this network.
          Regularization should be an addition to the cost function, while
          dropout should be added after both variable scopes.
    Hint: You might consider using a tensorflow Graph Collection (e.g
          "total_loss") to collect the regularization and loss terms (which you
          will add in add_loss_op below).
    Hint: Here are the dimensions of the various variables you will need to
          create

          W:  (2*embed_size, hidden_size)
          b1: (hidden_size,)
          U:  (hidden_size, label_size)
          b2: (label_size)

    Args:
      x: tf.Tensor of shape (-1, 2*embed_size)
    Returns:
      output: tf.Tensor of shape (batch_size, label_size)
    """

    with tf.variable_scope('Layer1') as scope:
      W = tf.get_variable(
          'W', [2 * self.config.embed_size,
                self.config.hidden_size])
      b1 = tf.get_variable('b1', [self.config.hidden_size])
      h = tf.nn.tanh(tf.matmul(x, W) + b1)
      if self.config.l2:
          tf.add_to_collection('total_loss', 0.5 * self.config.l2 * tf.nn.l2_loss(W))

    with tf.variable_scope('Layer2') as scope:
      U = tf.get_variable('U', [self.config.hidden_size, self.config.label_size])
      b2 = tf.get_variable('b2', [self.config.label_size])
      y = tf.matmul(h, U) + b2
      if self.config.l2:
          tf.add_to_collection('total_loss', 0.5 * self.config.l2 * tf.nn.l2_loss(U))
    output = tf.nn.dropout(y, self.dropout_placeholder)

    return output 

  def add_loss_op(self, y):
    """Adds cross_entropy_loss ops to the computational graph.

    Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
          implementation. You might find tf.reduce_mean useful.
    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y, self.labels_placeholder))
    tf.add_to_collection('total_loss', cross_entropy)
    loss = tf.add_n(tf.get_collection('total_loss'))

    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """

    optimizer = tf.train.AdamOptimizer(self.config.lr)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
  
  def __init__(self, config):
    self.config = config
    self.load_data(debug=True)
    self.add_placeholders()

    y = self.add_model(self.input_placeholder)

    self.loss = self.add_loss_op(y)
    self.predictions = tf.nn.softmax(y)
    one_hot_prediction = tf.argmax(self.predictions, 1)
    correct_prediction = tf.equal(
        tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
    self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
    self.train_op = self.add_training_op(self.loss)
    
  def run_epoch(self, session, input_data, input_labels,
                shuffle=True, verbose=50):
    orig_X, orig_y = input_data, input_labels
    dp = self.config.dropout
    
    total_loss = []
    total_correct_examples = 0
    total_processed_examples = 0
    total_steps = len(orig_X) / self.config.batch_size
    for step, (x, y) in enumerate(
      data_iterator(orig_X, orig_y, batch_size=self.config.batch_size,
                   label_size=self.config.label_size, shuffle=shuffle)):
      feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
      loss, total_correct, _ = session.run(
          [self.loss, self.correct_predictions, self.train_op],
          feed_dict=feed)
      total_processed_examples += len(x)
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
    return np.mean(total_loss), total_correct_examples / float(total_processed_examples)

  def predict(self, session, X, y=None):
    """Make predictions from the provided model."""
    # If y is given, the loss is also calculated
    # We deactivate dropout by setting it to 1
    dp = 1
    losses = []
    results = []
    if np.any(y):
        data = data_iterator(X, y, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    else:
        data = data_iterator(X, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    for step, (x, y) in enumerate(data):
      feed = self.create_feed_dict(input_batch=x, dropout=dp)
      if np.any(y):
        feed[self.labels_placeholder] = y
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
    for i, tag in sorted(index_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print 'Tag: {} - P {:2.4f} / R {:2.4f}'.format(tag, prec, recall)

def calculate_confusion(config, predicted_indices, y_indices):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((config.label_size, config.label_size), dtype=np.int32)
    for i in xrange(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion
    
def test_Model():
  """Test model implementation.

  You can use this function to test your implementation of the Named Entity
  Recognition network. When debugging, set max_epochs in the Config object to 1
  so you can rapidly iterate.
  """
  config = Config()
  with tf.Graph().as_default():
    model = Model(config)

    init = tf.initialize_all_variables()
#    saver = tf.train.Saver()

    with tf.Session() as session:
      best_val_loss = float('inf')
      best_val_epoch = 0

      session.run(init)
      for epoch in xrange(config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()
        ###
        train_loss, train_acc = model.run_epoch(session, model.X_train,
                                                model.y_train)
        val_loss, predictions = model.predict(session, model.X_dev, model.y_dev)
        print 'Training loss: {}'.format(train_loss)
        print 'Training acc: {}'.format(train_acc)
        print 'Validation loss: {}'.format(val_loss)
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_val_epoch = epoch
#          if not os.path.exists("./weights"):
#            os.makedirs("./weights")
#        
#          saver.save(session, './weights/ner.weights')
        if epoch - best_val_epoch > config.early_stopping:
          break
        ##
        confusion = calculate_confusion(config, predictions, model.y_dev)
        print_confusion(confusion, model.index_to_tag)
        print 'Total time: {}'.format(time.time() - start)
      
#      saver.restore(session, './weights/ner.weights')

if __name__ == "__main__":
    test_Model()
    
