from __future__ import print_function

import argparse
import numpy as np
import re

from collections import Counter, namedtuple
from itertools import chain, product
from time import time

np.seterr(divide='ignore', invalid='ignore')

LABELS = [
    'B-PER', 'B-ORG', 'B-LOC', 'B-MISC',
    'I-PER', 'I-ORG', 'I-LOC', 'I-MISC',
    'O',
    ]


class NER:
    Token = namedtuple('Token', ['word', 'pos', 'chunk', 'y'])

    def __init__(
            self, train_fn, dev_fn, test_fn=None, T=20, feat_freq=0,
            test_increment=0, labels=LABELS, identifier='',
    ):
        self.T = int(T)                  # number of perceptron iterations
        self.feat_freq = int(feat_freq)  # frequency threshold of features
        self.labels = labels

        # filekeeping...
        self.train_fn = train_fn
        self.identifier = identifier

        # during training, if `test_increment` is N, the model will evaluate
        # its performance (via `self.test`) on the training set after every
        # N iterations
        self.test_increment = int(test_increment)

        # the special start token!
        self.start_tok = NER.Token('<s>', 'BOS', 'BOS', 'BOS')

        self.label_idx = {}  # label to index
        self.idx_label = {}  # index to label

        self.feat_idx = {}  # feature to index
        self.idx_feat = {}  # index to feature

        # this is container for maping feature combinations (words, POS tags,
        # NER labels, etc.) to their feature indices in `self.featurize`
        self.features = {}

        # train the model
        self.train()

        # evaluate the model on the training set
        if not test_increment or T % test_increment != 0:
            self.test(train_fn)

        # evaluate the model on the dev set
        if dev_fn:
            self.test(dev_fn)

        # evaluate the model on the test set
        if test_fn:
            self.test(test_fn)

    # training ----------------------------------------------------------------

    def train(self):
        ''' '''
        print('Training...', end='\r')

        sents = self.load(self.train_fn)

        # extract gold NER labels if none were provided
        if not self.labels:
            self.labels = set([tok.y for tok in chain(*sents)])

        # index NER labels
        for i, label in enumerate(self.labels):
            self.label_idx[label] = i
            self.idx_label[i] = label

        self.L = i + 1  # number of unique labels

        # index gold NER labels
        gold = [[self.label_idx[tok.y] for tok in sent] for sent in sents]

        # collect and index features
        for i, feat in enumerate(self.collect_features(sents)):
            self.feat_idx[feat] = i
            self.idx_feat[i] = feat

        self.F = i + 1  # number of unique features

        # apply the structured perceptron algorithm to learn parameters
        self.perceptron(sents, gold)

    def collect_features(self, sents):
        ''' '''
        features = Counter()

        for sent in sents:
            prev, prev_y = self.start_tok, self.start_tok.y

            for tok in sent:
                y = self.label_idx[tok.y]
                features.update(NER._featurize(tok, prev, y, prev_y))
                prev, prev_y = tok, y

        if self.feat_freq:
            features = [f for f, c in features.items() if c >= self.feat_freq]

        return features

    def perceptron(self, sents, gold, train_fn=None):
        ''' '''
        # initialize weight vectors `alpha` and `gamma`
        self.alpha, gamma = np.zeros(self.F), np.zeros(self.F)

        nT = 0

        # iterate...
        for T in range(1, self.T + 1):
            print('Training... T=%d' % T, end='\r')
            status_quo = self.alpha.copy()

            for sent, Y in zip(sents, gold):
                prediction = self.viterbi(sent)

                if Y != prediction:

                    for feat_idx in self.global_phi(sent, Y):
                        self.alpha[feat_idx] += 1

                    for feat_idx in self.global_phi(sent, prediction):
                        self.alpha[feat_idx] -= 1

                    gamma += self.alpha
                    nT += 1

            # cease training if the parameters are no longer changing
            if np.array_equal(status_quo, self.alpha):
                print('\nConvergence at T=%d' % T)
                break

            # save the model's predictions at given intervals
            if self.test_increment and T % self.test_increment == 0:
                print('Testing at T=%d    ' % T, end='\r')
                alpha = self.alpha.copy()       # copy for safekeeping
                self.alpha = gamma / nT         # temporarily take the average
                self.test(self.train_fn, T=T)   # test on the training set
                self.alpha = alpha              # restore `self.alpha`
                del alpha

        else:
            print('\nTraining complete!')

        # average the weight vector
        self.alpha = gamma / nT

    # decoding ----------------------------------------------------------------

    def viterbi(self, sent):
        ''' '''
        N = len(sent)

        V = np.zeros((self.L, 1))       # viterbi vector
        B = np.full((self.L, N), -1)    # backtrace matrix

        # initialization
        word, prev, prev_y = sent[0], self.start_tok, self.start_tok.y

        for y in range(self.L):
            V[y] = self.inner_product(word, prev, y, prev_y)

        # recursion step
        prev = word

        for i, word in enumerate(sent[1:], start=1):
            transitions = np.repeat(V, self.L, axis=1)

            for prev_y, y in product(range(self.L), range(self.L)):
                transitions[prev_y, y] = self.inner_product(word, prev, y, prev_y)  # noqa

            V = transitions.max(0)[:, np.newaxis]
            B[:, i] = transitions.argmax(0)

            prev = word

        # backtrace
        prediction = [V.argmax(), ]

        for i in range(N - 1, -1, -1):
            prediction.append(B[prediction[-1], i])

        prediction = prediction[-2::-1]

        return prediction

    def inner_product(self, tok, prev, y, prev_y):
        ''' '''
        return sum(self.alpha[f] for f in self.local_phi(tok, prev, y, prev_y))

    # features ----------------------------------------------------------------

    def global_phi(self, sent, labels):
        ''' '''
        prev, prev_y = self.start_tok, self.start_tok.y

        for tok, y in zip(sent, labels):
            for feat_idx in self.local_phi(tok, prev, y, prev_y):
                yield feat_idx

                prev, prev_y = tok, y

    def local_phi(self, *args):
        ''' '''
        return self.features.setdefault(args, tuple(self.featurize(*args)))

    def featurize(self, *args):
        ''' '''
        for feat in NER._featurize(*args):

            try:
                yield self.feat_idx[feat]

            except KeyError:
                continue

    @staticmethod
    def _featurize(tok, prev, y, prev_y):
        ''' '''
        is_init = int(prev_y == 'BOS')
        is_upper = int(tok.word[0].isupper())

        return (
            # current features
            'w=%s' % tok.word,      # word
            'p=%s' % tok.pos,       # POS
            'c=%s' % tok.chunk,     # chunk
            'y=%s' % y,             # NER label

            # previous features
            '<w=%s' % prev.word,
            '<p=%s' % prev.pos,
            '<c=%s' % prev.chunk,
            '<y=%s' % prev_y,

            # bigram features
            '<w+w=%s_%s' % (prev.word, tok.word),
            '<p+p=%s_%s' % (prev.pos, tok.pos),
            '<c+c=%s_%s' % (prev.chunk, tok.chunk),
            '<y+y=%s_%s' % (prev_y, y),

            # current conjoined features
            'w+p=%s_%s' % (tok.word, tok.pos),
            'w+c=%s_%s' % (tok.word, tok.chunk),
            'p+c=%s_%s' % (tok.pos, tok.chunk),
            'w+p+c=%s_%s_%s' % (tok.word, tok.pos, tok.chunk),

            # previous conjoined features
            '<w+<p=%s_%s' % (prev.word, prev.pos),
            '<w+<c=%s_%s' % (prev.word, prev.chunk),
            '<p+<c=%s_%s' % (prev.pos, prev.chunk),
            '<w+<p+<c=%s_%s_%s' % (prev.word, prev.pos, prev.chunk),

            # contingent on current NER label ---------------------------------

            # current features
            '%s:w=%s' % (y, tok.word),
            '%s:p=%s' % (y, tok.pos),
            '%s:c=%s' % (y, tok.chunk),

            # bigram features
            '%s:<w+w=%s_%s' % (y, prev.word, tok.word),
            '%s:<p+p=%s_%s' % (y, prev.pos, tok.pos),
            '%s:<c+c=%s_%s' % (y, prev.chunk, tok.chunk),

            # current conjoined features
            '%s:w+p=%s_%s' % (y, tok.word, tok.pos),
            '%s:w+c=%s_%s' % (y, tok.word, tok.chunk),
            '%s:p+c=%s_%s' % (y, tok.pos, tok.chunk),
            '%s:w+p+c=%s_%s_%s' % (y, tok.word, tok.pos, tok.chunk),

            # previous conjoined features
            '%s:<w+<p=%s_%s' % (y, prev.word, prev.pos),
            '%s:<w+<c=%s_%s' % (y, prev.word, prev.chunk),
            '%s:<p+<c=%s_%s' % (y, prev.pos, prev.chunk),
            '%s:<w+<p+<c=%s_%s_%s' % (y, prev.word, prev.pos, prev.chunk),

            # contingent on previous NER label --------------------------------

            # current features
            '<%s:w=%s' % (prev_y, tok.word),
            '<%s:p=%s' % (prev_y, tok.pos),
            '<%s:c=%s' % (prev_y, tok.chunk),

            # bigram features
            '<%s:<w+w=%s_%s' % (prev_y, prev.word, tok.word),
            '<%s:<p+p=%s_%s' % (prev_y, prev.pos, tok.pos),
            '<%s:<c+c=%s_%s' % (prev_y, prev.chunk, tok.chunk),

            # current conjoined features
            '<%s:w+p=%s_%s' % (prev_y, tok.word, tok.pos),
            '<%s:w+c=%s_%s' % (prev_y, tok.word, tok.chunk),
            '<%s:p+c=%s_%s' % (prev_y, tok.pos, tok.chunk),
            '<%s:w+p+c=%s_%s_%s' % (prev_y, tok.word, tok.pos, tok.chunk),

            # previous conjoined features
            '<%s:<w+<p=%s_%s' % (prev_y, prev.word, prev.pos),
            '<%s:<w+<c=%s_%s' % (prev_y, prev.word, prev.chunk),
            '<%s:<p+<c=%s_%s' % (prev_y, prev.pos, prev.chunk),
            '<%s:<w+<p+<c=%s_%s_%s' % (prev_y, prev.word, prev.pos, prev.chunk),  # noqa

            # pseudo-linguistic features --------------------------------------

            # word length
            'len=%s' % len(tok.word),
            '%s:len=%s' % (y, len(tok.word)),

            # word initial
            'is_init=%s' % is_init,
            '%s:is_init=%s' % (y, is_init),
            '<%s:is_init=%s' % (prev_y, is_init),

            # capitalization
            '%s:is_upper=%s' % (y, is_upper),
            '<%s:is_upper=%s' % (prev_y, is_upper),

            # word-initial + capitalization
            '%s:is_upper+is_init=%s_%s' % (y, is_init, is_upper),
            '<%s:is_upper+is_init=%s_%s' % (prev_y, is_init, is_upper)
        )

    # test --------------------------------------------------------------------

    def test(self, data_fn, T=None):
        ''' '''
        if not T:
            print('Testing on %s' % data_fn)

        output = []

        for sent in iter(self.load(data_fn)):

            for tok, y in zip(sent, self.viterbi(sent)):
                y = self.idx_label[y]
                output.append('%s %s\n' % (' '.join(tok), y))

            output.append('\n')

        # for personal use...
        if self.identifier:
            data_fn += '.%s' % self.identifier

        with open('%s.%d.out' % (data_fn, T or self.T), 'w+') as f:
            for line in output:
                f.write(line)

    # utitlies ----------------------------------------------------------------

    @staticmethod
    def load(data_fn):
        ''' '''
        with open(data_fn, 'rb+') as f:
            text = f.read().decode('utf-8')

        data = []

        for sent in re.split(r'\n{2,}', text):
            sent = re.split(r'\n', sent)

            try:
                data.append([NER.Token(*w.split()) for w in sent])

            except TypeError:
                break

        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # arguments to instantiate NER()
    parser.add_argument('train_fn')
    parser.add_argument('dev_fn')
    parser.add_argument('--test_fn', '-t', type=str, default='')
    parser.add_argument('--T', '-T', type=int, default=20)
    parser.add_argument('--feat_freq', '-f', type=int, default=0)
    parser.add_argument('--test_increment', '-i', type=int, default=0)
    parser.add_argument('--all_labels', '-a', action='store_true')
    parser.add_argument('--identifier', '-id', type=str, default='')

    args = parser.parse_args()

    start = time()

    # instantiate the model
    NER(
        train_fn=args.train_fn,
        dev_fn=args.dev_fn,
        test_fn=args.test_fn,
        T=args.T,
        feat_freq=args.feat_freq,
        test_increment=args.test_increment,
        labels=LABELS if args.all_labels else None,
        identifier=args.identifier,
        )

    end = time()

    # print the time it took to train and test the model
    print(round((end - start) / 60, 4), 'minutes')
