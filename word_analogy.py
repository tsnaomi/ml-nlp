#!/usr/bin/python

import numpy as np
import os
import re


class WordAnalogy:

    def __init__(self, vector_fn, input_dir, output_dir, normalize, cosine):
        self.normalize = bool(normalize)
        self.cosine = bool(cosine)
        self.sim = self._cosine if cosine else self._euclidean
        self.word_vec = {}
        self.vec_word = {}
        self.vectors = []
        self.vectorize(vector_fn)
        self.test(input_dir, output_dir)

    def vectorize(self, vector_fn):
        '''Extract vectors from vector_fn as a 2-dimensional array.'''
        with open(vector_fn, 'rb') as f:
            vectors = filter(None, re.split(r'\s*[\r\n]+', f.read()))

        for i, vector in enumerate(vectors):
            vector = re.findall(r'(\S+)', vector)
            word, vector = vector[0], [float(n) for n in vector[1:]]
            self.vectors.append(vector)
            self.word_vec[word] = i
            self.vec_word[i] = word

        self.unk_vec = np.zeros((len(vector)))  # represent unknown words
        self.vectors = np.array(self.vectors)

        assert len(self.vectors.shape) == 2, 'Vectors are not homogeneous.'

        if self.normalize:
            self._normalize()

        if self.cosine:
            self.norms = self._Z(self.vectors)

    def test(self, input_dir, output_dir):
        '''Test the word analogy algorithm on input files.'''
        CORRECT = 0.
        TOTAL = 0.

        for fn in sorted(os.listdir(input_dir)):
            input_fn = os.path.join(input_dir, fn)
            output_fn = os.path.join(output_dir, fn)

            with open(input_fn, 'rb') as f:
                lines = filter(None, re.split(r'\s*[\r\n]+', f.read()))

            correct = 0.
            total = len(lines)
            TOTAL += total

            with open(output_fn, 'wb') as f:
                for line in lines:
                    A, B, C, D1 = re.split(r'\s+', line)
                    D2 = self.analogize(A, B, C)
                    f.write('%s %s %s %s\n' % (A, B, C, D2))
                    correct += D1 == D2

            CORRECT += correct
            acc = (correct / total) * 100

            print '%s:' % fn
            print 'ACCURACY TOP1: %.2f%% (%i/%i)' % (acc, correct, total)

        ACC = (CORRECT / TOTAL) * 100

        print '\nTotal accuracy: %.2f%% (%i/%i)' % (ACC, CORRECT, TOTAL)

    # normalization -----------------------------------------------------------

    def _normalize(self):
        '''Normalize the values in self.vectors, yielding unit vectors.'''
        self.vectors = (self.vectors.T / self._Z(self.vectors)).T

    def _Z(self, vectors):
        '''Calculate the normalization divisor.'''
        try:
            return np.sqrt(np.sum(vectors ** 2, 1))

        except ValueError:
            return np.sqrt(np.sum(vectors ** 2))

    # similarity --------------------------------------------------------------

    def analogize(self, A, B, C):
        '''Given A, B, and C, find D such that A to B is most like C to D.'''
        return self.vec_word[
            self.sim(self._get_vec(B) - self._get_vec(A) + self._get_vec(C))
            .argmax(0)
            ]

    def _cosine(self, Y):
        '''Calculate the cosine similarities between self.vectors and Y.'''
        return np.dot(self.vectors, Y) / (self.norms * self._Z(Y))

    def _euclidean(self, Y):
        '''Calculate the Euclidean distances between self.vectors and Y.'''
        return 0 - np.sqrt(np.sum((self.vectors - Y) ** 2, 1))

    def _get_vec(self, word):
        '''Retrieve the vector for a word (otherwise, the UNK vector).'''
        try:
            return self.vectors[self.word_vec[word]]

        except KeyError:
            return self.unk_vec


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_fn', '-v')
    parser.add_argument('--input_dir', '-i')
    parser.add_argument('--output_dir', '-o')
    parser.add_argument('--normalize', '-n')
    parser.add_argument('--cosine', '-c')
    args = parser.parse_args()

    WordAnalogy(
        args.vector_fn,
        args.input_dir,
        args.output_dir,
        int(args.normalize),
        int(args.cosine),
        )
