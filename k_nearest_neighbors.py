#!usr/bin/python

from __future__ import division

import numpy as np
import re

from collections import Counter


class kNN:

    def __init__(self, train_fn, test_fn, k, sim_func, sys_output_fn):
        self.train_fn = train_fn
        self.k = k
        self.sys_output = []
        self.sys_output_fn = sys_output_fn

        if sim_func == 1:
            self.sim_measure = 'Euclidean'
            self.sim = self.euclidean

        elif sim_func == 2:
            self.sim_measure = 'Cosine'
            self.sim = self.cosine

        else:
            raise ValueError('Please specify either 1 or 2 for similarity.')

        self.train()
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
        self.N = len(labels)  # number of training docs

        features = set(re.findall(r'(?<= )[\w-]+', docs))
        self.F = len(features)  # number of unique features

        self.idx_label = {}
        self.label_idx = {}
        self.feat_idx = {}

        for i, label in enumerate(self.labels):
            self.label_idx[label] = i
            self.idx_label[i] = label

        for i, feat in enumerate(features):
            self.feat_idx[feat] = i

        docs = filter(None, re.split(r'\s*[\r\n]+', docs))
        self.vectors, self.gold_labels = self.vectorize(docs)

    def vectorize(self, docs):
        '''Vectorize and return `docs` and their labels.'''
        n = len(docs)
        vectors = np.zeros((n, self.F))
        gold_labels = np.zeros(n, dtype=int)

        for i, doc in enumerate(docs):
            label, features = doc.split(' ', 1)
            gold_labels[i] = self.label_idx[label]

            for feat in features.split():
                feat, c = feat.split(':')

                try:
                    vectors[i, self.feat_idx[feat]] += int(c)

                except KeyError:
                    pass

        return vectors, gold_labels

    # test --------------------------------------------------------------------

    def test(self, test_fn, descriptor):
        '''Test the classifier on `test_fn`.'''
        with open(test_fn, 'r+') as f:
            docs = filter(None, re.split(r'\s*[\r\n]+', f.read()))

        self.sys_output.append('%%%%%%%%%% %s data:' % descriptor)
        self.confusion = np.zeros((self.L, self.L))

        test_vectors, gold_labels = self.vectorize(docs)
        sim = self.sim(test_vectors).argsort(1)[:, -self.k:]

        for i, vector in enumerate(sim):
            labels = Counter({label: 0 for label in self.labels})
            labels.update(self.idx_label[self.gold_labels[j]] for j in vector)
            probs = [(v / self.k, k) for k, v in labels.iteritems()]
            # sort by probability, then label:
            probs.sort(key=lambda t: (-t[0], t[1]))  # probs.sort(reverse=True)
            top_label = probs[0][1]
            self.confusion[gold_labels[i], self.label_idx[top_label]] += 1
            self.sys_output.append('array:%i %s %s' % (
                i, top_label, ' '.join('%s %.5f' % (l, p) for p, l in probs)))

        self.sys_output.append('\n')
        self.accuracy(descriptor)

    def euclidean(self, test_vectors):
        '''Calculate the Euclidean distances between test and train vectors.'''
        return 0 - (
            # MARRY ME:
            # https://medium.com/dataholiks-distillery/l2-distance-matrix-vectorization-trick-26aa3247ac6c
            -2 * np.dot(test_vectors, self.vectors.T)
            + np.sum(self.vectors ** 2, axis=1)
            + np.sum(test_vectors ** 2, axis=1)[:, np.newaxis]
            )

    def cosine(self, test_vectors):
        '''Calculate the cosine similarities between test and train vectors.'''
        return np.dot(test_vectors, self.vectors.T) / (
            np.sqrt(np.sum(self.vectors ** 2, 1))
            * np.sqrt(np.sum(test_vectors ** 2, 1)[:, np.newaxis])
            )

    def accuracy(self, descriptor):
        '''Print the model's classification accuracy and confusion matrix.'''
        print 'Confusion matrix for the %s data:' % descriptor
        print 'row is the truth, column is the system output\n'
        print '\t\t%s' % ' '.join(self.labels)

        for i, label in enumerate(self.labels):
            print label, ' '.join('%i' % i for i in self.confusion[i])

        accuracy = np.sum(self.confusion.diagonal()) / np.sum(self.confusion)

        print '\n\t%s accuracy=%.5f\n\n' % (descriptor.title(), accuracy)

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
    parser.add_argument('--k_val', '-k', type=int)
    parser.add_argument('--sim_func', '-sim', type=int)
    parser.add_argument('--sys_output_fn')
    args = parser.parse_args()

    kNN(
        train_fn=args.train_fn,
        test_fn=args.test_fn,
        k=args.k_val,
        sim_func=args.sim_func,
        sys_output_fn=args.sys_output_fn,
        )
