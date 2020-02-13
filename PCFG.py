#!/usr/bin/python

from __future__ import division

import nltk
import numpy as np
import re

from collections import defaultdict, Counter
from sys import stderr


class PCFG:
    '''This thing induces a PCFG given a treebank.'''

    def __init__(self, treebank_fn):
        self.productions = Counter()
        self.PCFG = defaultdict(lambda: defaultdict(float))
        self.load_treebank(treebank_fn)
        self.learn_probabilities()
        self.write()

    def load_treebank(self, treebank_fn):
        '''Load the productions deriving the trees in `treebank_fn`.'''
        with open(treebank_fn, 'rb') as f:
            treebank = filter(None, re.split(r'\s*[\r\n]+', f.read()))

        for tree in treebank:
            tree = nltk.Tree.fromstring(tree)
            self.productions.update(tree.productions())

        self.start = tree.productions()[0].lhs()

    def learn_probabilities(self):
        '''Assign probabilities to the grammar's productions.'''
        for rule, count in self.productions.iteritems():
            self.PCFG[rule.lhs()][rule.rhs()] += count

        for lhs, RHS in self.PCFG.iteritems():
            total = sum(RHS.itervalues())

            for rhs in RHS:
                self.PCFG[lhs][rhs] /= total

    def write(self):
        '''Print out the PCFG.'''
        LHS = sorted(self.PCFG.iterkeys())
        start = LHS.pop(LHS.index(self.start))
        LHS = [start] + LHS

        for lhs in LHS:
            for rhs, prob in self.PCFG[lhs].iteritems():
                if len(rhs) == 2:  # nonterminal
                    rhs = ' '.join([str(i) for i in rhs])

                elif '\'' not in rhs[0]:  # terminal
                    rhs = '\'%s\'' % rhs[0]

                else:  # terminal
                    rhs = '"%s"' % rhs[0]

                print '%s -> %s [%.10f]' % (str(lhs), rhs, prob)


class Parser:
    '''This is a PCFG parser.'''

    def __init__(self, grammar_fn, sentences_fn, smooth):
        self.smooth = smooth
        self.load_grammar(grammar_fn)
        self.parse_sentences(sentences_fn)

    def load_grammar(self, grammar_fn):
        '''Load the PCFG from `grammar_fn`.'''
        grammar = nltk.data.load(grammar_fn)
        self.start = str(grammar.start())
        self.terminals = defaultdict(set)
        self.nonterminals = defaultdict(set)
        heads = set()

        for rule in grammar.productions():
            p = np.log(rule.prob())
            lhs = str(rule.lhs())

            if rule.is_lexical():
                rhs = str(rule.rhs()[0])
                self.terminals[rhs].add((lhs, p))
                heads.add(lhs)

            else:
                rhs = rule.rhs()
                rhs = '%s %s' % (str(rhs[0].symbol()), str(rhs[1].symbol()))
                self.nonterminals[rhs].add((lhs, p))

        self.terminals = dict(self.terminals)
        self.nonterminals = dict(self.nonterminals)

        if self.smooth:
            self.model_unks(heads)

    def model_unks(self, heads):
        ''' '''
        p = 1. / len(heads)
        self.unks = [(head, p) for head in heads]

    def parse_sentences(self, sentences_fn):
        '''Read in and parse the sentences in `sentences_fn`.'''
        with open(sentences_fn, 'rb') as f:
            sentences = filter(None, re.split(r'\s*[\r\n]+', f.read()))

        for n, sent in enumerate(sentences):
            print self.PCKY(n, nltk.word_tokenize(sent))

    def PCKY(self, n, sentence):
        '''Parse the given sentence using PCKY.'''
        L = len(sentence) + 1
        P = [[''] * L for i in range(L)]

        try:
            # terminal rules
            for j, word in enumerate(sentence):
                P[j][j+1] = self.find_heads(word)

            # nonterminal rules
            for j in range(2, L):
                for i in range(j - 1)[::-1]:
                    for k in range(i + 1, j):
                        B = P[i][k]
                        C = P[k][j]
                        A = self.find_A(B, C)

                        try:
                            P[i][j].extend(A)

                        except AttributeError:
                            P[i][j] = A

            trees = [tree for tree in P[0][j] if tree.root == self.start]

            return max(trees, key=lambda t: t.prob).parse()[1:]  # fencepost

        except AttributeError:
            print >> stderr, 'Sentence (%i) encountered an OOV word.' % n

        except ValueError:
            print >> stderr, 'Grammar cannot derive Sentence (%i).' % n

        return ''

    def find_heads(self, word):
        '''Find all of the heads that dominate `word.`'''
        try:
            heads = self.terminals[word]

        except KeyError:
            heads = self.unks

        word = Tree(word, is_word=True)

        return [Tree(lhs, rhs=word, p=prob) for lhs, prob in heads]

    def find_A(self, B, C):
        '''Find A given B and C, where A -> B C.'''
        A = []

        for b in B:
            for c in C:
                parents = self.nonterminals.get('%s %s' % (b.root, c.root), [])

                for a, prob in parents:
                    A.append(Tree(a, lhs=b, rhs=c, p=prob + b.prob + c.prob))

        return A


class Tree:
    '''This is a probabilistic binary tree. Woot.'''

    def __init__(self, root=None, rhs=None, lhs=None, p=0, is_word=False):
        self.root = root
        self.lhs = lhs
        self.rhs = rhs
        self.prob = p
        self.is_word = is_word

    def __repr__(self):
        return self.root

    def parse(self):
        '''Return the string representation of the tree's parse.'''
        return ''.join(i for i in self.pre_order())

    def pre_order(self):
        '''Traverse and yield the tree's nodes in pre-order.'''
        if not self.is_word:
            yield ' ('

        else:
            yield ' '

        yield self.root

        if self.lhs:
            for n in self.lhs.pre_order():
                yield n

        if self.rhs:
            for n in self.rhs.pre_order():
                yield n

        if not self.is_word:
            yield ')'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--treebank_fn', '-t')
    parser.add_argument('--pcfg_fn', '-p')
    parser.add_argument('--sentences_fn', '-s')
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--task', choices=['induce', 'parse'])
    args = parser.parse_args()

    if args.task == 'induce':
        PCFG(args.treebank_fn)

    else:
        Parser(args.pcfg_fn, args.sentences_fn, args.smooth)
