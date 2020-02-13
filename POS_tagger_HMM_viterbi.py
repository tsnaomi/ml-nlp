from __future__ import print_function

import argparse
import json
import numpy as np
import re

from collections import defaultdict
from itertools import product
from time import time

try:
    from tabulate import tabulate

except ModuleNotFoundError:
    tabulate = None

np.seterr(divide='ignore', invalid='ignore')


class HMM:

    def __init__(
            self, train_fn, test_fn=None, n=3, lambdas=[],
            transition_k=0, emission_k=0, weight_k=False,
            smart_UNK_freq=0, UNK_by_length=0, UNK_freq=0,
            eval_train=False,
    ):
        self.n = max(2, int(n))  # bigram HMM at minimum
        self._n = self.n - 1

        # determine UNK-ing procedure
        self.UNK_freq = np.ceil(UNK_freq)
        self.smart_UNK_freq = np.ceil(smart_UNK_freq)
        self.UNK_by_length = int(UNK_by_length)

        # set interpolation weights for smoothing transition probabilities
        if lambdas:
            assert len(lambdas) == self.n and sum(lambdas) == 1, \
                'Invalid interpolation weights.'

        self.lambdas = list(lambdas)

        # set add-k values for smoothing transition and emission probabilities
        self.transition_k = float(transition_k)
        self.emission_k = float(emission_k)
        self.weight_k = weight_k

        self._set_UNK_methods()

        # train the model
        self.train(train_fn)

        self._describe_model()

        # evaluate the model on the training set
        if eval_train:
            self.test(train_fn)

        # evaluate the model on the test set
        if test_fn:
            self.test(test_fn)

    def _set_UNK_methods(self):
        ''' '''
        if self.smart_UNK_freq or self.UNK_by_length:
            self._get_sym_idx = self._get_sym_or_smart_UNK_idx

            if self.smart_UNK_freq and self.UNK_by_length:
                self._UNK_sym = self._UNK_by_morph_and_length

            elif self.smart_UNK_freq:
                self._UNK_sym = self._UNK_by_morph

            else:
                self._UNK_sym = self._UNK_by_length

        else:
            self._get_sym_idx = self._get_sym_or_UNK_idx

    def _describe_model(self):
        ''' '''
        description = '%s-gram' % self.n

        if self.lambdas:
            description += '  lambdas=%s' % str(self.lambdas)

        if self.emission_k:
            description += '  *' if self.weight_k else '  '
            description += 'emission-k=%s' % self.emission_k

        if self.transition_k:
            description += '  transition-k=%s' % self.transition_k

        if self.UNK_freq:
            description += '  UNK<=%i' % self.UNK_freq

            if self.UNK_by_length:
                description += '  UNK-len<=%i' % self.UNK_by_length

        if self.smart_UNK_freq:
            description += '  *UNK<=%i' % self.smart_UNK_freq

        print(description)

        return description

    # train -------------------------------------------------------------------

    def train(self, data_fn):
        ''' '''
        # collect transition and emission counts
        symbols, transitions, emissions = self._count(data_fn)

        # index symbols and states
        self._index_vocab(symbols)
        self._index_states(transitions)

        # instantiate transition and emission matrices
        self._vectorize_transitions(transitions)
        self._vectorize_emissions(emissions)

        # interpolate
        if self.lambdas:
            self._interpolate()

        # log and simplify matters
        with np.errstate(divide='ignore'):
            self.emissions = np.log2(self.emissions)
            self.transitions = np.log2(self.transitions[self._n])

        # get START and STOP transition probabilities
        self._vectorize_start_and_stop()

        # NOTE
        self.transitions = self.transitions[:-self.S, :-self.S]
        self.emissions = self.emissions[:-self.S, :]
        self.init_vec = self.init_vec[:-self.S]
        self.stop_vec = self.stop_vec[:-self.S]
        self.N -= self.S
        self.T -= 1

    def _count(self, data_fn):
        ''' '''
        with open(data_fn, 'rb+') as f:
            tweets = [json.loads(t) for t in f.readlines()]

        # collect symbol counts: {sym: count, }
        symbols = defaultdict(int)

        # collect state transitions: {n: {from_state: {to_state: count, }}}
        transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # noqa

        # collect emissions: {tag: {sym: count, }}
        emissions = defaultdict(lambda: defaultdict(int))

        for tweet in tweets:
            tweet = [('<s>', '|'), ] * self._n + tweet + [('</s>', '|'), ]

            # collect main counts
            for i in range(self._n, len(tweet)):
                sym, tag = tweet[i]
                symbols[sym] += 1

                # collect transition counts
                for n in range(self.n):
                    transition = tuple(t for _, t in tweet[i - n: i + 1])
                    from_state = transition[:-1] or None
                    to_state = transition[-self._n:] or (tag, )
                    transitions[n][from_state][to_state] += 1

                # collect emission counts
                emissions[tag][sym] += 1

        symbols = dict(symbols)
        transitions = dict(transitions)
        emissions = dict(emissions)

        return symbols, transitions, emissions

    def _index_vocab(self, symbols):
        ''' '''
        if self.UNK_freq or self.smart_UNK_freq:
            vocab = []

            for w, c in symbols.items():
                if c <= self.smart_UNK_freq:
                    if not self.__UNK_by_morph(w) and c > self.UNK_freq:
                        vocab.append(w)

                elif c > self.UNK_freq:
                    vocab.append(w)

            if self.smart_UNK_freq:
                vocab.extend([
                    '<UNK-#>', '<UNK-@>', '<UNK-U>', '<UNK-ed>', '<UNK-N>',
                    '<UNK-poss>', '<UNK-s>', '<UNK-L>', '<UNK-A>',
                    '<UNK-R>', '<UNK-$>', '<UNK-,>'
                    ])

            if self.UNK_by_length:
                for i in range(1, self.UNK_by_length + 1):
                    vocab.append('<UNK-%d>' % i)

        else:
            # include all of the training words contained in the vocabulary
            vocab = list(symbols.keys())

        vocab.remove('</s>')
        vocab.append('<UNK>')
        vocab.append('</s>')  # force STOP token to the end of the vocab

        self.V = len(vocab)     # vocabulary size
        self.sym_idx = {}       # sym to index
        self.idx_sym = {}       # index to sym

        # index emitted symbols
        for i, sym in enumerate(vocab):
            self.sym_idx[sym] = i
            self.idx_sym[i] = sym

        # set UNK index
        self.UNK_idx = i - 1

    def _index_states(self, transitions):
        ''' '''
        self.tags = [t[0] for t in transitions[0][None].keys()]
        self.tags.sort(key=lambda x: (x == '|', x[::-1]))

        states = {}

        # collect tag / unigram states
        states[0] = self.tags.copy()

        # collect bigram states
        states[1] = [tuple(i) for i in states[0]]

        # collect higher order n-gram states
        for n in range(2, self.n):
            states[n] = list(product(self.tags, repeat=n))
            states[n].sort(key=lambda x: (x[-1] == '|', x[::-1]))

        # map states to indices
        self.state_idx = {}                              # state to index
        self.idx_state = {n: {} for n in range(self.n)}  # index to state
        self.state_tag_idx = {}                          # state to tag index

        for n in range(self.n):
            for i, state in enumerate(states[n]):
                self.state_idx[state] = i       # state to index
                self.idx_state[n][i] = state    # index to state

                # map the state index to the index of its state-final tag
                self.state_tag_idx[i] = self.state_idx[state[-1]]

        # get the index of the START tag
        self.init_idx = self.state_idx[('|', ) * self._n]

        self.T = len(self.tags)  # number of unique tags
        self.N = i + 1           # number of states for the highest order ngram
        self.S = int(len(states[n]) / self.T)  # number of states per tag

    def _vectorize_transitions(self, transitions):
        ''' '''
        self.transitions = {}

        # instantiate unigram vector
        self.transitions[0] = np.zeros((self.T, 1), dtype=np.float64)

        # map unigram transition counts to vector
        to_states = transitions[0][None]
        to_idxs = [self.state_idx[t] for t in to_states.keys()]
        self.transitions[0][to_idxs, 0] = list(to_states.values())

        # convert transition counts to unigram probabilities
        self.transitions[0] /= self.transitions[0].sum()

        for n in range(1, self.n):

            # instantiate transition vectors
            N = len(self.idx_state[n])
            self.transitions[n] = np.zeros((N, self.N), dtype=np.float64)

            # map transition counts to vectors
            for from_state, to_states in transitions[n].items():
                from_idx = self.state_idx[from_state]
                to_idxs = [self.state_idx[t] for t in to_states.keys()]
                self.transitions[n][from_idx, to_idxs] = \
                    list(to_states.values())

            # perform add-k smoothing on transition counts
            if n == self._n and self.transition_k:
                self._k_smooth_transitions()

            # convert transition counts to maximum likelihood estimates
            self.transitions[n] /= self.transitions[n].sum(1)[:, np.newaxis]
            self.transitions[n][np.isnan(self.transitions[n])] = 0

    def _vectorize_emissions(self, emissions):
        ''' '''
        # instantiate emission vectors
        self.emissions = np.zeros((self.T, self.V), dtype=np.float64)

        # map emission counts to vectors
        for to_state, syms in emissions.items():
            state_idx = self.state_idx[to_state]

            for sym, count in syms.items():
                sym_idx = self._get_sym_idx(sym)
                self.emissions[state_idx, sym_idx] = count

        # delete STOP token probabilies
        self.emissions = self.emissions[:, :-1]
        self.V -= 1

        # perform add-k smoothing on emission counts
        if self.emission_k:

            # weight the add-k value per emission distribution
            if self.weight_k:
                self.kw = np.zeros((self.T, 1), dtype=np.float64)

                for i, (to_state, syms) in enumerate(emissions.items()):
                    state_idx = self.state_idx[to_state]
                    self.kw[state_idx] = len(syms)

                self.kw[self.T - 1] = 0  # ignore STOP probabilities
                self.kw /= self.kw.sum()
                self.emissions += self.kw * self.emission_k

            else:
                self.emissions += self.emission_k

        # convert emission counts to maximum likelihood estimates
        self.emissions /= self.emissions.sum(1)[:, np.newaxis]
        self.emissions[np.isnan(self.emissions)] = 0

        # NOTE
        if self.n > 2:
            self.emissions = np.repeat(self.emissions, self.S, axis=0)

    def _vectorize_start_and_stop(self):
        ''' '''
        # get transition-from-START probabilities
        self.init_vec = self.transitions[self.init_idx, np.newaxis].T

        # get transition-to-STOP probabilities
        if self.n == 2:
            self.stop_vec = self.transitions[:, self.init_idx][:, np.newaxis]

        else:
            self.stop_vec = np.empty((self.N, 1))

            for i, from_state in self.idx_state[self._n].items():
                for j, to_state in self.idx_state[self._n].items():
                    if to_state[-1] == '|':
                        self.stop_vec[i] = self.transitions[i, j]

    def _k_smooth_transitions(self):
        ''' '''
        if self.n == 2:
            self.transitions[self._n] += self.transition_k

        else:
            states = list(self.idx_state[self._n].items())
            start = [(self.init_idx, ('|', ) * self._n)]

            # only add k to possible transitions
            for i, from_state in states[:-self.S] + start:
                for j, to_state in states:
                    if from_state[1:] == to_state[:-1]:
                        self.transitions[self._n][i, j] += self.transition_k

        # don't smooth START-to-STOP transition
        self.transitions[self._n][self.init_idx, self.init_idx] -= \
            self.transition_k

    def _interpolate(self):
        ''' '''
        for n in range(self.n):
            self.transitions[n] *= self.lambdas[n]

            for i in range(len(self.transitions[n])):

                try:
                    from_state = self.idx_state[n][i]
                    backoff = from_state[1:] or from_state[0]
                    backoff_idx = self.state_idx[backoff]
                    self.transitions[n][i, :] += \
                        self.transitions[n - 1][backoff_idx, :]

                except KeyError:
                    break

    # test --------------------------------------------------------------------

    def test(self, data_fn):
        ''' '''
        with open(data_fn, 'rb+') as f:
            tweets = [json.loads(t) for t in f.readlines()]

        confusion = np.zeros((self.T, self.T), dtype=int)

        for tweet in tweets:
            words, gold = [], []

            for word, tag in tweet:
                words.append(word)
                gold.append(self.state_idx[tag])

            predicted = self.viterbi(words)

            for i, j in zip(predicted, gold):
                confusion[i, j] += 1

        acc = confusion.diagonal().sum() / confusion.sum()

        print('acc:   %s (%s)' % (np.round(acc, 5), data_fn))
        # print(self.tabulate(confusion, latex=True))

        return acc, confusion

    def viterbi(self, tweet):
        ''' '''
        T = len(tweet)

        V = self.init_vec.copy()        # viterbi vector
        B = np.full((self.N, T), -1)    # backtrace matrix

        # initialization
        x = self._get_sym_idx(tweet[0])
        V += self.emissions[:, x][:, np.newaxis]
        B[:, 0] = 0

        # recursion step
        for i, word in enumerate(tweet[1:], start=1):
            x = self._get_sym_idx(word)
            m = self.transitions + self.emissions[:, x] + V
            V = m.max(0)[:, np.newaxis]
            B[:, i] = m.argmax(0)

        V += self.stop_vec

        # backtrace
        tags = [V.argmax(), ]

        for i in range(T - 1, -1, -1):
            tags.append(B[tags[-1], i])

        tags = [self.state_tag_idx[t] for t in tags[-2::-1]]

        return tags

    def decipher(self, tag_idxs):
        ''' '''
        return [self.idx_state[0][t] for t in tag_idxs]

    def tag(self, tweet):
        ''' '''
        return self.decipher(self.viterbi(tweet))

    def tabulate(self, confusion, latex=False):
        ''' '''
        if tabulate:
            table = [[t, ] + list(r) for t, r in zip(self.tags, confusion)]
            format_ = 'latex' if latex else 'fancy_grid'

            return tabulate(table, headers=self.tags[:-1], tablefmt=format_)

    # utilities ---------------------------------------------------------------

    def _get_sym_or_UNK_idx(self, sym):
        ''' '''
        return self.sym_idx.get(sym, self.UNK_idx)

    def _get_sym_or_smart_UNK_idx(self, sym):
        ''' '''
        try:
            return self.sym_idx[sym]

        except KeyError:
            return self.sym_idx[self._UNK_sym(sym)]

    def _UNK_by_morph_and_length(self, sym):
        ''' '''
        return self.__UNK_by_morph(sym) or self._UNK_by_length(sym)

    def _UNK_by_length(self, sym):
        ''' '''
        return '<UNK-%d>' % min(len(sym), self.UNK_by_length)

    def _UNK_by_morph(self, sym):
        ''' '''
        return self.__UNK_by_morph(sym) or '<UNK>'

    def __UNK_by_morph(self, sym):  # noqa
        ''' '''
        sym = sym.lower()

        # if the token begins with hashtag...
        if sym.startswith('#'):
            return '<UNK-#>'

        # if the token begins with '@'...
        if sym.startswith('@'):
            return '<UNK-@>'

        # if the token begins with 'http', 'www', or matches a weak email
        # regex...
        if sym.startswith(('http', 'www') or re.match(r'^[\w\d.+-]+@[\w\d-.]+\.[\w\d-]+', sym, flags=re.I)):  # noqa
            return '<UNK-U>'

        # if the token ends in 'ed' (e.g., 'played')...
        if sym.endswith('ed'):
            return '<UNK-ed>'

        # if the token ends in a common noun suffix...
        if sym.endswith(('tion', 'ment', 'ness', 'ity', 'ism')):
            return '<UNK-N>'

        # if the token takees a common possessive ending...
        if sym.endswith(("'s", "s'")):
            return '<UNK-poss>'

        # if the token ends in 's'...
        if sym.endswith('s'):
            return '<UNK-s>'

        # if the token ends in a common contraction...
        if sym.endswith(("'ve", "'re", "'ll", "'a", "'d")):
            return '<UNK-L>'

        # if the token ends in a common adjectival suffix...
        if sym.endswith(('ible', 'able', 'ful', 'al', 'ic', 'ive', 'less', 'ous', 'ish')):  # noqa
            return '<UNK-A>'

        # if the token ends in 'ly' (e.g., 'friendly')...
        if sym.endswith('ly'):
            return '<UNK-R>'

        # if the token contains numbers...
        if re.search(r'\d+', sym):
            return '<UNK-$>'

        # if the token contains no alphanumeric characters...
        if not re.search(r'\w+', sym, flags=re.I):
            return '<UNK-,>'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # arguments to instantiate HMM()
    parser.add_argument('train_fn')
    parser.add_argument('--test_fn', '-t')
    parser.add_argument('--n', '-n', type=int, default=3)
    parser.add_argument('--lambdas', '-i', nargs='*', type=float, default=[])
    parser.add_argument('--emission_k', '-ek', type=float, default=0.)
    parser.add_argument('--transition_k', '-tk', type=float, default=0.)
    parser.add_argument('--weight_k', '-wk', action='store_true')
    parser.add_argument('--UNK_freq', '-u', type=int, default=0)
    parser.add_argument('--UNK_by_length', '-l', type=int, default=0)
    parser.add_argument('--smart_UNK_freq', '-b', type=int, default=0)
    parser.add_argument('--eval_train', '-tt', action='store_true')

    args = parser.parse_args()

    start = time()

    # instantiate the model
    HMM(
        train_fn=args.train_fn,
        test_fn=args.test_fn,
        n=args.n,
        lambdas=args.lambdas,
        emission_k=args.emission_k,
        transition_k=args.transition_k,
        weight_k=args.weight_k,
        UNK_freq=args.UNK_freq,
        UNK_by_length=args.UNK_by_length,
        smart_UNK_freq=args.smart_UNK_freq,
        eval_train=args.eval_train,
        )

    end = time()

    # print the time it took to train and test the model
    print(round((end - start) / 60, 4), 'minutes')
