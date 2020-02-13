#!/usr/bin/python

from __future__ import division

import numpy as np
import re

from sys import stderr

np.seterr(all='raise')


class POSTagger:
    '''Where MaxEnt meets Beam Search.'''

    def __init__(self, data_fn, boundary_fn, model_fn, B, N, K):
        self.data_fn = data_fn
        self.boundary_fn = boundary_fn
        self.model_fn = model_fn
        self.B = B  # beam width
        self.N = N  # top N
        self.K = K  # top K
        self._dtype = [('i', '<i8'), ('p', '<f8')]  # for safe keeping

        self.load_maxent_model()
        self.load_test_data()
        self.test()

    # load --------------------------------------------------------------------

    def load_maxent_model(self):
        '''Load the MaxEnt model weights and tags.'''
        with open(self.model_fn, 'rb+') as f:
            model = f.read()

        tags = re.findall(
            r'(?<=FEATURES FOR CLASS )\S+',
            model,
            flags=re.M,
            ) + ['BOS']
        self.T = len(tags)  # number of unique tags
        self._T = range(self.T)

        features = set(re.findall(r'(?<=^ )\S+', model, flags=re.M))
        self.F = len(features)  # number of unique features (incl. <default>)
        features.remove('<default>')

        self.tag_idx = {}
        self.idx_tag = {}
        self.feat_idx = {'<default>': 0}

        for i, tag in enumerate(tags):
            self.tag_idx[tag] = i
            self.idx_tag[i] = tag

        for i, feat in enumerate(features, start=1):
            self.feat_idx[feat] = i

        self.weights = np.zeros((self.T, self.F))

        class_features = filter(
            None,
            re.split(r'FEATURES FOR CLASS.*$', model, flags=re.M),
            )

        for tag_idx, feats in enumerate(class_features):
            for feat in re.split(r'\s*[\r\n]+', feats):
                try:
                    feat, w = feat.split()
                    self.weights[tag_idx, self.feat_idx[feat]] = float(w)

                except ValueError:
                    pass  # feat is an empty string

    def load_test_data(self):
        '''Load the sentences in `data_fn` as a 2D list using `boundary_fn`.'''
        with open(self.data_fn, 'r+') as f:
            sentences = filter(None, re.split(r'\s*[\r\n]+', f.read()))

        with open(self.boundary_fn, 'r+') as f:
            boundaries = re.split(r'\s*[\r\n]+', f.read())

        indices = []
        n = 0

        for i in boundaries:
            try:
                i = int(i)
                indices.append(n + i)
                n += i

            except ValueError:
                pass  # i is an empty string

        indices = zip([0] + indices, indices)
        self.sentences = [sentences[j:k] for j, k in indices]

    # tag ---------------------------------------------------------------------

    def test(self):
        '''Tag some sentences.'''
        correct = 0
        total = 0

        for sentence in self.sentences:
            sys_output = self.beam_tag(sentence)

            for (word, gold), (tag_idx, prob) in sys_output:
                tag = self.idx_tag[tag_idx]
                correct += gold == tag
                total += 1

                print '%s %s %s %.5f' % (word, gold, tag, prob)

        print >> stderr, '\nAccuracy (B=%i, N=%i, K=%i): %.5f' % (
            self.B,
            self.N,
            self.K,
            correct / total,
            )

    def beam_tag(self, sentence):
        '''Tag `sentence` via MaxEnt and bean search -- ehem, BEAM search.'''
        paths = np.zeros((1, len(sentence) + 3), dtype=self._dtype)
        paths[0, :2] = self.tag_idx['BOS'], 0
        sys_gold = []

        for i, word in enumerate(sentence, start=2):
            new_paths = []

            # split `word` into a list of features and partially vectorize it,
            # then retrieve its initial MaxEnt probabilities
            word = re.findall(r'\S+', word)
            sys_gold.append(word[:2])
            partial_probs = self.maxent_tag_1(self.vectorize(word[2::2]))

            for path in paths:
                # construct 'previous tags' features
                t1 = self.idx_tag[path[i - 2]['i']]
                t2 = self.idx_tag[path[i - 1]['i']]
                prev_tags = ['prevT=' + t2, 'prevTwoTags=%s+%s' % (t1, t2)]

                # get tag probabilities given the fully vectorized word
                probs = self.maxent_tag_2(partial_probs.copy(), prev_tags)
                tags = np.array(zip(self._T, probs), dtype=self._dtype)

                # select only the "top N" POS tags
                tags = tags[tags['p'].argsort()][-self.N:]

                # create new paths
                path = np.array([path] * self.N)
                path[:, i] = tags
                path[:, -1]['p'] += np.log10(tags['p'])
                new_paths.append(path)

            # prune new paths, adhering to the beam width and keeping only the
            # "top K" paths
            paths = np.concatenate(new_paths, axis=0)
            max_prob = path[:, -1]['p'].max()
            paths = paths[paths[:, -1]['p'].argsort()][-self.K:, :]
            paths = paths[np.where(paths[:, -1]['p'] + self.B >= max_prob)]

        # woot! extract and return the best predicted tag sequence
        tags = paths[-1, 2:-1].tolist()

        return zip(sys_gold, tags)

    def maxent_tag_1(self, word_vector):
        '''Begin to calculate MaxEnt tag probabilities given `word_vector`.

        This method returns the dot product between the `word_vector` and
        feature weights, excluding the features `prevT` and `prevTwoTags`.
        '''
        return np.sum(word_vector * self.weights, 1)

    def maxent_tag_2(self, probs_vector, prev_tags):
        ''''Return the MaxEnt probability of each tag given `probs_vector`.

        This method completes calculating the MaxEnt probability of each tag
        given the `word_vector` provided to self.maxent_tag_1(). This method
        incorporates the weights (if any) for the features `prevT` and
        `prevTwoTags`, determined according to `prev_tags`
        '''
        # incorporate `prevT=` and `prevTwoTags` features
        for feat in prev_tags:
            try:
                probs_vector += self.weights[:, self.feat_idx[feat]]

            except KeyError:
                pass

        # finish calculating MaxEnt probabilities
        probs_vector = np.exp(probs_vector)
        probs_vector /= probs_vector.sum()  # normalize by Z

        return probs_vector

    def vectorize(self, word_features):
        '''Vectorize `word_features`.'''
        vector = np.zeros(self.F)
        vector[0] = 1  # allows the <default> value to come into play

        for feat in word_features:
            try:
                vector[self.feat_idx[feat]] = 1

            except KeyError:
                pass

        return vector

    # -------------------------------------------------------------------------


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fn')
    parser.add_argument('--boundary_fn')
    parser.add_argument('--model_fn')
    parser.add_argument('--beam_size', type=int)
    parser.add_argument('--N', type=int)
    parser.add_argument('--K', type=int)
    args = parser.parse_args()

    POSTagger(
        data_fn=args.data_fn,
        boundary_fn=args.boundary_fn,
        model_fn=args.model_fn,
        B=args.beam_size,
        N=args.N,
        K=args.K,
        )
