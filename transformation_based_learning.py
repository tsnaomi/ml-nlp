#!usr/bin/Python

from __future__ import division

import numpy as np
import re

from sys import stderr


class TBL:
    '''Transformation-based Error-Driven Learning.'''

    def __init__(self, train_fn, test_fn, model_fn, min_gain=1, N=1):
        if min_gain and min_gain < 0:
            raise ValueError('The value of `min_gain` must be greater than 0.')

        if not (train_fn or model_fn):
            raise ValueError('Please provide `train_fn` or `model_fn`.')

        self.train_fn = train_fn
        self.test_fn = test_fn
        self.model_fn = model_fn
        self.min_gain = min_gain
        self.N = N

        # train...
        if train_fn:
            self.train()

        # test...
        if test_fn:
            self.test()

    # train -------------------------------------------------------------------

    def train(self):
        '''Train the TBL model -- i.e., learn transformations.'''
        self.vectorize(self.train_fn)
        self.transformations = []
        net_gain, feat_idx, from_idx, to_idx = float('inf'), None, None, None

        print self.idx_label[0]  # print `model_fn` data on the fly

        while self.min_gain <= net_gain:

            try:
                # print `model_fn` data on the fly
                print '%s %s %s %i' % (
                    self.idx_feat[feat_idx],
                    self.idx_label[from_idx],
                    self.idx_label[to_idx],
                    net_gain,
                    )

                # transform self.vectors!!!!
                self.transform(feat_idx, from_idx, to_idx)

            except KeyError:
                # if feat_idx, from_idx, and to_idx are None
                pass

            net_gains = []

            for from_idx in self.label_vec:
                docs = self.vectors[self.vectors[:, -1] == from_idx]

                for to_idx in self.label_vec:

                    if to_idx == from_idx:
                        continue

                    gains = docs[docs[:, -2] == to_idx, :-2].sum(0)
                    gains -= docs[docs[:, -2] == docs[:, -1], :-2].sum(0)
                    feat_idx = gains.argmax()
                    net_gain = gains[feat_idx]
                    net_gains.append((net_gain, feat_idx, from_idx, to_idx))

            net_gain, feat_idx, from_idx, to_idx = max(net_gains)
            self.transformations.append((feat_idx, from_idx, to_idx))

        # pop the last transformation, which did not yield a sufficient enough
        # net_gain
        self.transformations.pop()

    # test --------------------------------------------------------------------

    def test(self):
        '''Test the TBL model on the data in self.test_fn.'''
        self.vectorize(self.test_fn)

        if self.model_fn:
            self.load_model()

        self.apply_transformations()

        for i, doc in enumerate(self.vectors):
            sys_output = 'array:%i %s %s' % (
                i,
                self.idx_label[doc[-2]],
                self.idx_label[doc[-1]],
                )

            for feat_idx, from_idx, to_idx in self.traces[i]:
                sys_output += ' %s %s %s' % (
                    self.idx_feat[feat_idx],
                    self.idx_label[from_idx],
                    self.idx_label[to_idx],
                    )

            print sys_output

        accuracy = (self.vectors[:, -2] == self.vectors[:, -1]).sum() / self.D

        print >> stderr, '\nAccuracy (N=%i): %.5f' % (self.N, accuracy)

    def load_model(self):
        '''Load the TBL model (i.e., transformations) from self.model_fn.'''
        with open(self.model_fn, 'rb+') as f:
            model = filter(None, re.split(r'\s*[\r\n]+', f.read()))

        self.vectors[:, -1] = self.label_idx[model[0]]  # initial annotation
        self.transformations = []

        for transformation in model[1:self.N + 1]:
            feat, from_label, to_label = transformation.split()[:3]

            try:
                self.transformations.append((
                    self.feat_idx[feat],
                    self.label_idx[from_label],
                    self.label_idx[to_label],
                    ))

            except KeyError:
                pass

    # utilties ----------------------------------------------------------------

    def vectorize(self, data_fn):
        '''Vectorize the data in `data_fn`, producing self.vectors.'''
        with open(data_fn, 'rb+') as f:
            docs = f.read()

        labels = re.findall(r'(?<=^)[\S]+', docs, flags=re.M)
        self.labels = list(set(labels))  # document labels
        self.L = len(self.labels)  # number of unique labels
        self.D = len(labels)  # number of training docs

        features = sorted(list(set(re.findall(r'(?<= )[\w-]+', docs))))
        self.F = len(features)  # number of unique features

        self.label_idx = {}
        self.idx_label = {}
        self.feat_idx = {}
        self.idx_feat = {}

        for i, label in enumerate(self.labels):
            self.label_idx[label] = i
            self.idx_label[i] = label

        for i, feat in enumerate(features):
            self.feat_idx[feat] = i
            self.idx_feat[i] = feat

        self.label_vec = np.array(self.idx_label.keys())

        # the final two columns for self.vectors are for the gold and current
        # test labels (respectively); since self.vectors is initialized with
        # zeros, the initial annotation is done implicitly, since the label
        # that appears first in `data_fn` is represented/indexed as '0'
        self.vectors = np.zeros((self.D, self.F + 2))

        for i, doc in enumerate(re.split(r'\s*[\r\n]+', docs)):
            if doc:
                label, features = doc.split(' ', 1)
                self.vectors[i, -2] = self.label_idx[label]  # gold label

                for feat in re.findall(r'\S+(?=:)', features):
                    self.vectors[i, self.feat_idx[feat]] = 1

    def apply_transformations(self):
        '''Apply self.transformations' transformations to self.vectors.'''
        self.traces = [[] for i in range(self.D)]

        for transformation in self.transformations:
            indices = self.transform(*transformation)

            for i in indices:
                self.traces[i].append(transformation)

    def transform(self, feat_idx, from_idx, to_idx):
        '''Label the documents with `feat_idx` and `from_idx` to `to_idx`.'''
        indices = np.intersect1d(
            np.where(self.vectors[:, -1] == from_idx),
            np.where(self.vectors[:, feat_idx] == 1)
            )
        self.vectors[indices, -1] = to_idx

        return indices

    # -------------------------------------------------------------------------


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fn', nargs='?')
    parser.add_argument('--test_fn', nargs='?')
    parser.add_argument('--model_fn', nargs='?')
    parser.add_argument('--min_gain', nargs='?', default=1, type=int)
    parser.add_argument('--N', nargs='?', default=1, type=int)
    args = parser.parse_args()

    TBL(
        train_fn=args.train_fn,
        test_fn=args.test_fn,
        model_fn=args.model_fn,
        min_gain=args.min_gain,
        N=args.N,
        )
