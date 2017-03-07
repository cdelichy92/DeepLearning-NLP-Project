import json
import numpy as np
from collections import defaultdict

DATA_PATH = 'snli_1.0/'
LABEL_TO_INDEX = {'contradiction':0,
                   'entailment':1,
                   'neutral':2}
INDEX_TO_LABEL = {v:k for k,v in LABEL_TO_INDEX.items()}

class Vocab():

    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.total_words = 0 # total number of words parsed
        self.word_freq = defaultdict(int)
        self.padding = '<pad>'
        self.unknown = '<unk>'
        self._add_word(self.padding, count=0)
        self._add_word(self.unknown, count=0)
        self.embedding_matrix = None


    def _add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def construct(self, words):
        for word in words:
            self._add_word(word)
        self.total_words = sum(self.word_freq.values())
        print('{} total words parsed and {} unique words'.format(self.total_words, len(self.word_freq)))


    def encode(self, word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        else:
            return self.word_to_index[self.unknown]


    def decode(self, index):
        return index_to_word[index]


    def __len__(self):
        return len(self.word_freq)


    def build_embedding_matrix(self, dim):
        if dim == 100:
            glove_file = 'glove/glove.6B/glove.6B.100d.txt'
        else:
            glove_file = 'glove/glove.840B.300d.txt'
        self.embedding_matrix = np.zeros((self.__len__(), dim))
        for line in open(glove_file, 'r'):
            dat = line.split(' ')
            word = dat[0]
            if word in self.word_to_index:
                embedding = np.array(dat[1:], dtype='float32')
                self.embedding_matrix[self.word_to_index[word]] = embedding

        self.embedding_matrix[self.word_to_index[self.unknown]] = \
            -2*np.ones(dim, dtype='float32')


def get_words_dataset(dataset='train'):
    fp = DATA_PATH + 'snli_1.0_{}.jsonl'.format(dataset)
    for line in open(fp, 'r'):
        data = json.loads(line)
        data_parsed = data['sentence1'].strip().split()
        data_parsed += data['sentence2'].strip().split()
        for word in data_parsed:
            yield clean(word)


def clean(string):
    string = ( string.replace('.','')
                 .replace(',', '')
                 .replace('!', '')
                 .replace('?', '')
                 .replace('"', '') )
    try:
        # sometimes some symbols are left at the beginning/end of words
        # (e.g. - / )
        string = string[:] if string[0].isalpha() else string[1:]
        string = string[:] if string[-1].isalpha() else string[:-1]
    except IndexError:
        pass
    string = string.lower()
    return string


def pad_sentence(vocab, sentence, target_len, padding):
    if len(sentence) > target_len:
            sentence = sentence[0:target_len]
    else:
        fix = [vocab.encode(vocab.padding)]*(target_len-len(sentence))
        sentence = fix + sentence if padding == 'pre' else sentence + fix
    return sentence


def encode_sentence(vocab, sentence):
    sent = clean(sentence).split()
    encoded_sent = [vocab.encode(word) for word in sent]
    return encoded_sent


def get_sentences_dataset( vocab, target_len, dataset='train', padding='pre'):
    fp = DATA_PATH + 'snli_1.0_{}.jsonl'.format(dataset)
    for line in open(fp):
        data_line = json.loads(line)
        if "gold_label" not in data_line:
            continue
        sent1 = clean(data_line["sentence1"]).split()
        sent2 = clean(data_line["sentence2"]).split()
        try:
            label = LABEL_TO_INDEX[data_line["gold_label"]]
            if label == '-':
                continue
        except:
            continue
        sent1 = [vocab.encode(word) for word in sent1]
        sent2 = [vocab.encode(word) for word in sent2]
        len1, len2 = min(len(sent1), target_len), min(len(sent2), target_len)
        if len1 == 0:
            len1 = 1
            sent1 = [vocab.encode(vocab.unknown)]
        if len2 == 0:
            len2 = 1
            sent2 = [vocab.encode(vocab.unknown)]
        sent1 = pad_sentence(vocab, sent1, target_len, padding)
        sent2 = pad_sentence(vocab, sent2, target_len, padding)
        sent1 = np.array(sent1, dtype=np.int32)
        sent2 = np.array(sent2, dtype=np.int32)
        yield sent1, sent2, len1, len2, label


def data_iterator(orig_sent1, orig_sent2, orig_len1, orig_len2, orig_y=None, batch_size=32,
        label_size=3, shuffle=True):
    N = orig_sent1.shape[0]
    if shuffle:
        indices = np.random.permutation(N)
    else:
        indices = np.arange(N)
    data_sent1, data_sent2 = orig_sent1[indices,:], orig_sent2[indices,:]
    data_len1, data_len2 = orig_len1[indices], orig_len2[indices]
    data_y = orig_y[indices] if np.any(orig_y) else None
    n_batches = int(np.ceil(N / batch_size))
    for i in range(n_batches): # n_batches-1 so that all of the batches have batch_size size
        sent1 = data_sent1[i*batch_size:min((i+1)*batch_size, N), :]
        sent2 = data_sent2[i*batch_size:min((i+1)*batch_size, N), :]
        len1 = data_len1[i*batch_size:min((i+1)*batch_size, N)]
        len2 = data_len2[i*batch_size:min((i+1)*batch_size, N)]
        y = None
        if np.any(data_y):
            y_indices = data_y[i*batch_size:min((i+1)*batch_size, N)]
            y = np.zeros((len(y_indices), label_size), dtype=np.int32)
            y[np.arange(len(y_indices)), y_indices] = 1
        yield sent1, sent2, len1, len2, y


if __name__ == '__main__':
    vocab = Vocab()
    vocab.construct(get_words_dataset(dataset='train'))
    print(len(vocab))
    print(vocab.word_to_index)
