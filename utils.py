from collections import defaultdict

import numpy as np
import json

class Vocab(object):
  def __init__(self):
    self.word_to_index = {}
    self.index_to_word = {}
    self.word_freq = defaultdict(int)
    self.total_words = 0
    self.unknown = '<unk>'
    self.padding = '<pad>'
    self.add_word(self.unknown, count=0)
    self.add_word(self.padding, count=0)

  def add_word(self, word, count=1):
    if word not in self.word_to_index:
      index = len(self.word_to_index)
      self.word_to_index[word] = index
      self.index_to_word[index] = word
    self.word_freq[word] += count

  def construct(self, words):
    for word in words:
      self.add_word(word)
    self.total_words = float(sum(self.word_freq.values()))
    print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))

  def encode(self, word):
    if word not in self.word_to_index:
      word = self.unknown
    return self.word_to_index[word]

  def decode(self, index):
    return self.index_to_word[index]

  def __len__(self):
    return len(self.word_freq)

def get_snli_dataset(dataset='train'):
  fn = 'snli_1.0/snli_1.0_{}.jsonl'
  for line in open(fn.format(dataset)):
    data = json.loads(line)
    data_parsed = data["sentence1"].strip().split()
    data_parsed += data["sentence2"].strip().split()

    for word in data_parsed:
        yield clean(word)
      #yield word.replace('"','').replace('.','').replace(',','').lower()
      
def adjust_sentence_len(vocab, sentence, target_len):
    if len(sentence) > target_len:
            sentence=sentence[0:target_len]
    else:
        while len(sentence) < target_len:
            sentence.insert(0,vocab.encode(vocab.padding))
            #sentence.append(vocab.encode(vocab.padding))
    return sentence
    
def clean(text):
    return text.replace('"','').replace('.','').replace(',','').lower()
    
def get_snli_sentences(vocab, target_len, dataset='train'):
    
    #dataset could be "train", "test" or "dev"
    fn = 'snli_1.0/snli_1.0_{}.jsonl'
    
    for line in open(fn.format(dataset)):
        data_line = json.loads(line)
        if "gold_label" not in data_line:
            continue
#        sent1 = data_line["sentence1"].replace('.','').lower().split()
#        sent2 = data_line["sentence2"].replace('.','').lower().split()
        sent1 = clean(data_line["sentence1"]).split()
        sent2 = clean(data_line["sentence2"]).split()
        try:
            label = data_line["gold_label"].encode('utf-8')
            if label == '-':
                continue
        except:
            continue
        sent1 = [vocab.encode(word) for word in sent1]
        sent2 = [vocab.encode(word) for word in sent2]
        sent1 = adjust_sentence_len(vocab, sent1, target_len)
        sent2 = adjust_sentence_len(vocab, sent2, target_len)
        sent1 = np.array(sent1,dtype=np.int32)
        sent2 = np.array(sent2,dtype=np.int32)
        yield sent1,sent2,label

def get_features(tokens, wordVectors, sentence1, sentence2):
    """ Obtain the sentence feature for sentiment analysis by averaging its word vectors """
    # Implement computation for the sentence features given a sentence.                                                       
    
    # Inputs:                                                         
    # - tokens: a dictionary that maps words to their indices in    
    #          the word vector list                                
    # - wordVectors: word vectors (each row) for all tokens                
    # - sentence1: a list of words in the premise sentence
    # - sentence2: a list of words in the hypothesis sentence

    # Output:                                                         
    # - input_vec: feature vector for the classifier

    sent1_vec = np.zeros((wordVectors.shape[1],))
    indices1 = [tokens[word] for word in sentence1 if word in tokens]
    sent1_vec = np.mean(wordVectors[indices1, :], axis=0)
    
    sent2_vec = np.zeros((wordVectors.shape[1],))
    indices2 = [tokens[word] for word in sentence2 if word in tokens]
    sent2_vec = np.mean(wordVectors[indices2, :], axis=0)
    
    input_vec = np.concatenate((sent1_vec,sent2_vec), axis=1)
    
    return input_vec

def data_iterator(orig_X, orig_y=None, batch_size=32, label_size=3, shuffle=False):
  # Optionally shuffle the data before training
  if shuffle:
    indices = np.random.permutation(len(orig_X))
    data_X = orig_X[indices]
    data_y = orig_y[indices] if np.any(orig_y) else None
  else:
    data_X = orig_X
    data_y = orig_y
  ###
  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    x = data_X[batch_start:batch_start + batch_size]
    # Convert our target from the class index to a one hot vector
    y = None
    if np.any(data_y):
      y_indices = data_y[batch_start:batch_start + batch_size]
      y = np.zeros((len(x), label_size), dtype=np.int32)
      y[np.arange(len(y_indices)), y_indices] = 1
    ###
    yield x, y
    total_processed_examples += len(x)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)
  
def data_iterator_lstm(orig_sent1, orig_sent2, orig_y=None, batch_size=32, label_size=3, shuffle=False):
  # Optionally shuffle the data before training
  if shuffle:
    indices = np.random.permutation(orig_sent1.shape[0])
    data_sent1 = orig_sent1[indices,:]
    data_sent2 = orig_sent2[indices,:]
    data_y = orig_y[indices] if np.any(orig_y) else None
  else:
    data_sent1 = orig_sent1
    data_sent2 = orig_sent2
    data_y = orig_y
  ###
  total_steps = int(np.ceil(orig_sent1.shape[0] / float(batch_size)))
  for step in xrange(total_steps-1):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    sent1 = data_sent1[batch_start:batch_start + batch_size,:]
    sent2 = data_sent2[batch_start:batch_start + batch_size,:]
    # Convert our target from the class index to a one hot vector
    y = None
    if np.any(data_y):
      y_indices = data_y[batch_start:batch_start + batch_size]
      y = np.zeros((len(y_indices), label_size), dtype=np.int32)
      y[np.arange(len(y_indices)), y_indices] = 1
    ###
    yield sent1, sent2, y
  
if __name__ == "__main__":
    vocab = Vocab()
    vocab.construct(get_snli_dataset('dev'))
    counter = 1    
    for thing in get_snli_sentences(vocab, 25, dataset='dev'):
        counter+=1
        print thing
        if counter > 10:
            break

    print counter
