#!usr/bin/python

from __future__ import division

import numpy as np
import re

from collections import Counter


class NaiveBayes:

    def __init__(
        self, train_fn, test_fn, prior_delta, cond_delta, model_fn, output_fn,
        model='Bernoulli'
    ):
        self.model = model
        self.train_fn = train_fn
        self.prior_delta = prior_delta
        self.likelihood_delta = cond_delta
        self.model = []
        self.model_fn = model_fn
        self.sys_output = []
        self.sys_output_fn = output_fn

        if model.title() == 'Bernoulli':
            self.train_likelihoods = self.train_bernoulli_likelihoods
            self.score = self.score_bernoulli

        elif model.title() == 'Multinomial':
            self.train_likelihoods = self.train_multinomial_likelihoods
            self.score = self.score_multinomial

        else:
            raise ValueError('Please specify either Bernoulli or Multinomial.')

        self.train()
        self.create_model()
        self.test(train_fn, 'training')
        self.test(test_fn, 'test')
        self.create_sys_output()

    # train -------------------------------------------------------------------

    def train(self):
        '''Train the classifier.'''
        with open(self.train_fn, 'rb+') as f:
            docs = f.read()

        labels = re.findall(r'(?<=^)[\S]+', docs, flags=re.M)
        self.labels = list(set(labels))  # document labels
        self.L = len(self.labels)  # number of unique labels
        N = len(labels)  # number of training docs

        vocab = set(re.findall(r'(?<= )[\w-]+', docs))
        V = len(vocab)  # vocabulary size

        # indexing for indexing's sake...
        self.label_idx = {}
        self.term_idx = {}

        _label_counts = Counter(labels)
        label_counts = []

        for i, label in enumerate(self.labels):
            self.label_idx[label] = i
            label_counts.append(_label_counts[label])

        for i, term in enumerate(vocab):
            self.term_idx[term] = i

        # smoothed priors
        self.label_counts = np.array(label_counts).reshape((self.L, 1))
        self.priors = self.label_counts + self.prior_delta
        self.priors /= self.prior_delta * N + N

        # smoothed likelihoods
        self.likelihoods = np.zeros((self.L, V))
        self.train_likelihoods(docs)

    def train_bernoulli_likelihoods(self, docs):
        '''Calculate Bernoulli likelihood terms.'''
        for doc in re.split(r'\s*[\r\n]+', docs):
            if doc:
                label, doc = doc.split(' ', 1)
                label_idx = self.label_idx[label]

                for term in re.findall(r'[\w-]+(?=:)', doc):
                    self.likelihoods[label_idx, self.term_idx[term]] += 1

        den = self.likelihood_delta * 2 + self.label_counts
        self.likelihoods = (self.likelihoods + self.likelihood_delta) / den

        # smoothed zero-count likelihood
        self.zero_count_likelihood = self.likelihood_delta / den

    def train_multinomial_likelihoods(self, docs):
        '''Calculate Multinomial likelihood terms.'''
        for doc in re.split(r'\s*[\r\n]+', docs):
            if doc:
                label, doc = doc.split(' ', 1)
                label_idx = self.label_idx[label]

                for term in doc.split():
                    term, c = term.split(':')
                    self.likelihoods[label_idx, self.term_idx[term]] += int(c)

        den = np.sum(self.likelihoods + self.likelihood_delta, 1)
        den = den.reshape((self.L, 1))
        self.likelihoods = (self.likelihoods + self.likelihood_delta) / den

        # smoothed zero-count likelihood
        self.zero_count_likelihood = self.likelihood_delta / den

    # test --------------------------------------------------------------------

    def test(self, test_fn, descriptor):
        '''Test the classifier on `test_fn`.'''
        with open(test_fn, 'r+') as f:
            docs = filter(None, re.split(r'\s*[\r\n]+', f.read()))

        self.sys_output.append('%%%%%%%%%% %s data:' % descriptor)
        self.confusion = np.zeros((self.L, self.L))

        for i, doc in enumerate(docs):
            label, doc = doc.split(' ', 1)
            s = self.score(doc)  # get posterior probabilities
            self.sys_output.append('array:%i %s %s' % (
                i, label, ' '.join('%s %.5f' % (l, p) for p, l in s)))
            self.confusion[self.label_idx[label], self.label_idx[s[0][1]]] += 1

        self.sys_output.append('\n')
        self.accuracy(descriptor)

    def score_bernoulli(self, doc):
        '''Calculate the Bernoulli posterior of each label given `doc`.'''
        scores = []

        for label_idx in range(self.L):
            scores.append(np.log10(self.priors[label_idx][0]))

            for term in re.findall(r'[\w-]+(?=:)', doc):
                try:
                    p = self.likelihoods[label_idx, self.term_idx[term]]

                except KeyError:
                    p = self.zero_count_likelihood[label_idx][0]

                scores[-1] += np.log10(p) + np.log10(1 - p)

        #      sort the posterior probabilities from greatest to least
        return sorted(zip(scores, self.labels), reverse=True)

    def score_multinomial(self, doc):
        '''Calculate the Multinomial posterior of each label given `doc`.'''
        scores = []

        for label_idx in range(self.L):
            scores.append(np.log10(self.priors[label_idx][0]))

            for term in doc.split():
                term, c = term.split(':')

                try:
                    p = self.likelihoods[label_idx, self.term_idx[term]]

                except KeyError:
                    p = self.zero_count_likelihood[label_idx][0]

                scores[-1] += np.log10(p) * int(c)

        #      sort the posterior probabilities from greatest to least
        return sorted(zip(scores, self.labels), reverse=True)

    def accuracy(self, descriptor):
        '''Print the model's classification accuracy and confusion matrix.'''
        print 'Confusion matrix for the %s data:' % descriptor
        print 'row is the truth, column is the system output\n'
        print '\t\t%s' % ' '.join(self.labels)

        for i, label in enumerate(self.labels):
            print label, ' '.join('%i' % i for i in self.confusion[i])

        accuracy = np.sum(self.confusion.diagonal()) / np.sum(self.confusion)

        print '\n\t%s accuracy=%.5f\n\n' % (descriptor.title(), accuracy)

    # system files ------------------------------------------------------------

    def create_model(self):
        '''Write the model's prior and likelihood probabilities to file.'''
        # self.priors
        self.model.append('%%%%% prior prob P(c) %%%%%')

        for label, label_idx in self.label_idx.iteritems():
            p = self.priors[label_idx][0]
            self.model.append('%s\t%.5f\t%.5f' % (label, p, np.log10(p)))

        # self.likelihoods
        self.model.append('%%%%% conditional prob P(f|c) %%%%%')

        for label_idx, label in enumerate(self.labels):
            self.model.append(
                '%%%%%%%%%% conditional prob P(f|c) c=%s %%%%%%%%%%' % label)
            likelihoods = []

            for term, term_idx in self.term_idx.iteritems():
                p = self.likelihoods[label_idx, term_idx]
                likelihoods.append(
                    '%s\t%s\t%.5f\t%.5f' % (term, label, p, np.log10(p)))

            self.model.extend(sorted(likelihoods))

        with open(self.model_fn, 'w+') as f:
            f.write('\n'.join(self.model))

    def create_sys_output(self):
        '''Write the classification results to file.'''
        with open(self.sys_output_fn, 'w+') as f:
            f.write('\n'.join(self.sys_output))

    # -------------------------------------------------------------------------


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fn')
    parser.add_argument('--test_fn')
    parser.add_argument('--prior_delta', type=float)
    parser.add_argument('--cond_delta', type=float)
    parser.add_argument('--model_fn')
    parser.add_argument('--output_fn')
    parser.add_argument('--model', default='Bernoulli')
    args = parser.parse_args()

    NaiveBayes(
        train_fn=args.train_fn,
        test_fn=args.test_fn,
        prior_delta=args.prior_delta,
        cond_delta=args.cond_delta,
        model_fn=args.model_fn,
        output_fn=args.output_fn,
        model=args.model,
        )
