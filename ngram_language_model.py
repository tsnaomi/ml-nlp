from __future__ import print_function

import argparse
import numpy as np
import re

from collections import Counter, defaultdict
from time import time


np.seterr(divide='raise')


class LanguageModel:

    def __init__(self, train_fn, test_fn='', n=3, k=0, lambdas=[], OOV_freq=1):
        self.n = max(1, int(n))  # `n` must be at least 1
        self._n = self.n - 1

        self.k = float(k)  # add-k

        self.OOV_freq = max(1, int(np.ceil(OOV_freq)))  # force UNK modeling

        # validate interpolation weights
        if lambdas:

            if abs(1 - sum(lambdas)) > 10e-3:
                raise ValueError('Interpolation weights must sum to 1: %s.')

            if any(l < 0 for l in lambdas):
                raise ValueError('Interpolation weights must be >= to 0.')

            L = len(lambdas)

            if L != n:
                raise ValueError('The # of interpolation weights must = n.')

            if L == 1:
                lambdas = []

        # set interpolation weights (if any)
        self.lambdas = list(lambdas)

        # configure methods for later use
        self._set_methods()

        # train and test on the training set
        self.train(train_fn)
        train_PP = self.test(train_fn)

        # test on the test set (if any)
        if test_fn:
            test_PP = self.test(test_fn)

        # print results
        print(
            '\n%s-gram' % self.n,
            '\ntrain PP:      %s (%s)' % (train_PP, train_fn),
            '\ntest PP:       %s (%s)' % (test_PP, test_fn) if test_fn else '',
            '\nOOV freq:      %s' % self.OOV_freq,
            '\nadd-k:         %s' % self.k if self.k else '',
            '\nlambdas:       %s' % str(lambdas)[1:-1] if lambdas else '',
            '\n',
            )

    def _set_methods(self):
        '''Configure methods to instantiate and smooth vectors.'''
        if self.lambdas and self.k:
            self._unseen_prob = self._unseen_interp_k_prob
            self._instantiate_vectors = self._instantiate_k_vectors

        elif self.lambdas:
            self._unseen_prob = self._unseen_interp_prob
            self._instantiate_vectors = self._instantiate_0_vectors

        elif self.k:
            self._unseen_prob = self._unseen_k_prob
            self._instantiate_vectors = self._instantiate_k_vectors

        else:
            self._unseen_prob = self._unseen_unsmoothed_prob
            self._instantiate_vectors = self._instantiate_0_vectors

    # train -------------------------------------------------------------------

    def train(self, data_fn):
        '''Train the language model on the sentences in `data_fn`.'''
        counts = self._count(data_fn)

        # container for maximum likelihood estimates: {n: [history x word], }
        self.ngrams = {}

        for n in counts.keys():

            # instantiate vectors
            N = len(counts[n])  # number of histories of length `n`
            self.ngrams[n] = self._instantiate_vectors(N)

            for i, (history, words) in enumerate(counts[n].items()):

                # index history
                self.gram_idx[n][history] = i
                self.idx_gram[n][i] = history

                # map counts to vector
                self.ngrams[n][i, list(words.keys())] += list(words.values())

            # convert counts to maximum likelihood estimates
            for i in range(N):
                self.ngrams[n][i, :] /= self.ngrams[n][i, :].sum()

        # determine the k-smoothed maximum likelihood estimate of an ngram
        # with a history that was not observed in the training data
        if self.k and self.n > 1:
            self.__unseen_k_prob = self.k / (self.V * self.k)

        # interpolate maximum likelihood estimates
        if self.lambdas:
            self._interpolate()

    def _count(self, data_fn):
        '''Return a dictionary of the ngram counts in `data_fn`.'''
        with open(data_fn, 'r+') as f:
            sents = ' ' + f.read() + ' '

        # get initial unigram counts to determine the vocabulary and which
        # low-frequency words to treat as OOV items (i.e., '<UNK>')
        counts = Counter(re.findall(r'\S+', sents)).items()
        vocab = [w for w, c in counts if c > self.OOV_freq]
        vocab += ['<UNK>', '</s>']

        self.V = len(vocab)  # vocabulary size

        self.gram_idx = defaultdict(dict)  # ngram to index
        self.idx_gram = defaultdict(dict)  # index to ngram

        # index unigrams
        for i, word in enumerate(vocab):
            self.gram_idx[0][word] = i
            self.idx_gram[0][i] = word

        # set UNK index
        self.UNK_idx = self.gram_idx[0]['<UNK>']

        # split `sents` into a list of sentences
        sents = filter(None, re.split(r'\s*[\r\n]+', sents[1:-1]))

        # collect ngram counts: {n: {history: {word: count}, }, }
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for n, history, word_idx in self._iter_train(sents):
            counts[n][history][word_idx] += 1

        # convert defaultdicts to dictionaries (better for debugging)
        counts = \
            {k: {i: dict(j) for i, j in d.items()} for k, d in counts.items()}

        return counts

    def _interpolate(self):
        '''Interpolate the maximum likelihood estimates in `self.ngrams`.'''
        for n in range(self.n):
            self.ngrams[n] *= self.lambdas[n]

            for i in range(len(self.ngrams[n])):

                try:
                    history = self.idx_gram[n][i]
                    backoff = self._backoff(history)
                    backoff_idx = self.gram_idx[n - 1][backoff]
                    self.ngrams[n][i, :] += self.ngrams[n - 1][backoff_idx, :]

                except TypeError:
                    break

        # determine the interpolated probabilities of words conditioned
        # on histories that were not seen in training
        if self.k:
            # interpolate the k-smoothed probabilities of unseen ngrams
            unseen = np.array(self.lambdas) * self.__unseen_k_prob
            self.__unseen_interp_k_probs = \
                [j + unseen[i + 1:].sum() for i, j in enumerate(unseen)][1:]

        else:
            # re-normalize the interpolation weights for backoff
            for n in range(self._n):
                self.ngrams[n] /= sum(self.lambdas[:n + 1])

    def _iter_train(self, sents):
        '''Iterate across `sents`, yielding ngrams.'''
        if self.n == 1:

            # iterate across unigrams
            for sent in sents:
                for i, tok in enumerate(sent.split() + ['</s>', ]):
                    word_idx = self._get_word_idx(tok)[0]

                    yield 0, None, word_idx

        else:
            n = self._n
            onset = ('<s>', ) * n

            if self.lambdas:

                # iterate across unigrams, bigrams, etc. for interpolation
                for sent in sents:
                    history = onset

                    for i, tok in enumerate(sent.split() + ['</s>', ]):
                        word_idx, tok = self._get_word_idx(tok)

                        yield 0, None, word_idx

                        for n in range(1, self.n):
                            yield n, history[-n:], word_idx

                        history = history[1:] + (tok, )

            else:
                # iterate across grams of size `self.n`
                for sent in sents:
                    history = onset

                    for i, tok in enumerate(sent.split() + ['</s>', ]):
                        word_idx, tok = self._get_word_idx(tok)

                        yield n, history, word_idx

                        history = history[1:] + (tok, )

    def _instantiate_k_vectors(self, N):
        '''Instantiate training vectors with the add-k smoothing value.'''
        return np.full((N, self.V), self.k)

    def _instantiate_0_vectors(self, N):
        '''Instantiate training vectors with zeros.'''
        return np.zeros((N, self.V))  # faster to instantiate than np.full()

    # test --------------------------------------------------------------------

    def test(self, data_fn):
        '''Calculate the model's perplexity on the sentences in `data_fn`.'''
        M = 0
        P = 0

        for history, word_idx in self._iter_test(data_fn):
            M += 1

            try:
                P += np.log2(self._word_prob(history, word_idx))

            except FloatingPointError:
                return np.inf

        L = P / M
        PP = 2 ** -L

        return PP

    def _word_prob(self, history, word_idx):
        '''Return the probability of the word `word_idx` given `history`.'''
        try:
            hist_idx = self.gram_idx[self._n][history]

            return self.ngrams[self._n][hist_idx][word_idx]

        except KeyError:
            return self._unseen_prob(history, word_idx)

    def _iter_test(self, data_fn):
        '''Iterate across the sentences in `test_fn`, yielding ngrams.'''
        with open(data_fn, 'r+') as f:
            sents = filter(None, re.split(r'\s*[\r\n]+', f.read()))

        if self.n == 1:

            # iterate across unigrams
            for sent in sents:
                for i, tok in enumerate(sent.split() + ['</s>', ]):
                    word_idx = self._get_word_idx(tok)[0]

                    yield None, word_idx

        else:
            n = self._n
            onset = ('<s>', ) * n

            # iterate across grams of size `self.n`
            for sent in sents:
                history = onset

                for i, tok in enumerate(sent.split() + ['</s>', ]):
                    word_idx, tok = self._get_word_idx(tok)

                    yield history, word_idx

                    history = history[1:] + (tok, )

    # unseen probabilities ----------------------------------------------------

    def _unseen_interp_k_prob(self, history, word_idx=None):
        '''Return P(`word_idx`|`history`) when `history` is unseen.

        This method returns a k-smoothed, interpolated probability.
        '''
        n = self._n

        while history:
            n -= 1

            try:
                history = self._backoff(history)
                history_idx = self.gram_idx[n][history]
                seen_prob = self.ngrams[n][history_idx, word_idx]
                unseen_prob = self.__unseen_interp_k_probs[n]

                return seen_prob + unseen_prob

            except KeyError:
                pass

    def _unseen_interp_prob(self, history, word_idx):
        '''Return P(`word_idx`|`history`) when `history` is unseen.

        This method returns an interpolated probability.
        '''
        n = self._n

        while history:
            n -= 1

            try:
                history = self._backoff(history)
                history_idx = self.gram_idx[n][history]

                return self.ngrams[n][history_idx, word_idx]

            except KeyError:
                pass

    def _unseen_k_prob(self, history=None, word_idx=None):
        '''Return P(`word_idx`|`history`) when `history` is unseen.

        This method returns a k-smoothed probability.
        '''
        return self.__unseen_k_prob

    def _unseen_unsmoothed_prob(self, history=None, word_idx=None):
        '''Return P(`word_idx`|`history`) when `history` is unseen.

        This method returns an unsmoothed probability -- i.e., 0.
        '''
        return 0  # no smoothing whatsoever

    # probability distributions -----------------------------------------------

    def confirm_all_distributions(self, data_fn):
        '''Validate all of the ngram probability distributions in `data_fn`.'''
        histories = set()

        for history, word_idx in self._iter_test(data_fn):
            histories.add(history)

        copacetic = True

        for history in histories:

            try:
                self.confirm_distribution(history)

            except AssertionError as dist_sum:
                copacetic = False

                print('Invalid probability distribution given %s: SUM = %s' % (
                    str(history),
                    dist_sum,
                    ))

        if copacetic:
            print('Probability distributions OK!')

    def confirm_distribution(self, history):
        '''Validate the probability distribution conditioned on `history.'''
        dist_sum = sum(self._word_prob(history, i) for i in range(self.V))

        assert abs(1 - dist_sum) < 10e-3, dist_sum

    # utilities ---------------------------------------------------------------

    def _get_word_idx(self, tok):
        '''Return the index of `tok` (or the index of <UNK>).'''
        try:
            return self.gram_idx[0][tok], tok

        except KeyError:
            return self.UNK_idx, '<UNK>'  # replace OOV words with '<UNK>'

    def _backoff(self, history):
        '''Back off `history` by one word.'''
        return history[1:] or None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # arguments to instantiate LanguageModel()
    parser.add_argument('train_fn')
    parser.add_argument('--test_fn', '-t')
    parser.add_argument('--n', '-n', type=int, default=3)
    parser.add_argument('--k', '-k', type=float, default=0.)
    parser.add_argument('--lambdas', '-w', nargs='*', type=float, default=[])
    parser.add_argument('--OOV_freq', '-u', type=float, default=1.)

    # not an argument of LanguageModel()
    parser.add_argument('--confirm_fn', '-c')

    args = parser.parse_args()

    start = time()

    # instantiate the language model
    LM = LanguageModel(
        train_fn=args.train_fn,
        test_fn=args.test_fn,
        n=args.n,
        k=args.k,
        lambdas=args.lambdas,
        OOV_freq=args.OOV_freq,
        )

    end = time()

    # print the time it took to train and test the model
    print(round((end - start) / 60, 4), 'minutes')

    # if `confirm_fn` is given, validate all of the probability distributions
    # encountered in `confirm_fn`(NB: this takes a few hours...)
    if args.confirm_fn:
        LM.confirm_all_distributions(args.confirm_fn)

    del LM  # welp
