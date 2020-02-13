#!/usr/bin/python

import nltk

from collections import defaultdict
from warnings import warn


class Rule:

    def __init__(self, LHS, RHS=[]):
        self.LHS = LHS
        self.RHS = RHS

    def __repr__(self):
        return '%s -> %s' % (self.LHS, ' '.join(self.RHS))

    def __eq__(self, y):
        return str(self) == y

    def __hash__(self):
        return hash(str(self))

    def is_hybrid(self):
        '''Return True if the RHS contains borh terminalS AND non-terminals.'''
        return not self.is_terminal() and not self.is_non_terminal()

    def is_non_terminal(self):
        '''Return True if the RHS contains only non-terminals.'''
        return not any(r for r in self.RHS if self._is_terminal_sym(r))

    def is_terminal(self):
        '''Return True if the RHS is a single terminal symbol.'''
        return len(self.RHS) == 1 and self._is_terminal_sym(self.RHS[0])

    def _is_terminal_sym(self, sym):
        '''Return True if the given symbol is a terminal symbol.'''
        return sym.startswith(('\'', '"')) and sym.endswith(('\'', '"'))

    def is_unit(self):
        '''Return True if the rule is a unit rule.'''
        return self.is_non_terminal() and len(self.RHS) == 1

    def is_long(self):
        '''Return True if the RHS contains more than two symbols.'''
        return len(self.RHS) > 2


class CNF:

    def __init__(self, input_grammar):
        self.rules = []
        self.start = None
        self.load_productions(input_grammar)
        self.convert_to_CNF()

    def __repr__(self):
        return '\n'.join(str(rule) for rule in self.rules)

    def load_productions(self, input_grammar):
        '''Load the production rules from the given CFG grammar.'''
        grammar = nltk.data.load(input_grammar)
        self.start = str(grammar.start())

        for rule in grammar.productions():
            LHS, RHS = str(rule).split(' -> ')
            RHS = RHS.split()
            self.rules.append(Rule(LHS, RHS))

    def convert_to_CNF(self):
        '''Convert the CFG rules into Chomsky Normal Form.'''
        self._hybrid_productions()
        self._unit_productions()
        self._long_productions()
        self._clean()

        print self

    def _hybrid_productions(self):
        '''Eliminate hybrid rules.'''
        rules = []

        for rule in self.rules:
            if rule.is_hybrid():
                RHS = []

                for rhs in rule.RHS:
                    if rule._is_terminal_sym(rhs):
                        non_terminal_sym = '<' + rhs[1:-1] + '>'  # NOTE
                        RHS.append(non_terminal_sym)
                        rules.append(Rule(non_terminal_sym, [rhs]))

                    else:
                        RHS.append(rhs)

                rule.RHS = RHS

            rules.append(rule)

        self.rules = rules

    def _unit_productions(self):  # noqa
        '''Eliminate unit rules.'''
        def traverse_units(RHS):
            while RHS:
                rhs = RHS.pop()
                RHS.extend(unit_rules_dict[rhs])
                yield rhs

        rules_dict = defaultdict(list)
        rules = []
        unit_rules_dict = defaultdict(list)
        unit_rules = []

        for rule in self.rules:
            if rule.is_unit():
                unit_rules_dict[rule.LHS].append(rule.RHS[0])
                unit_rules.append(rule)

            else:
                rules_dict[rule.LHS].append(rule.RHS)
                rules.append(rule)

        while unit_rules:
            yardstick = []

            for rule in unit_rules:
                for LHS in traverse_units(list(rule.RHS)):
                    for RHS in rules_dict[LHS]:
                        new_rule = Rule(rule.LHS, RHS)
                        rules.append(new_rule)
                        rules_dict[new_rule.LHS].append(new_rule.RHS)

                for RHS in rules_dict[rule.RHS[0]]:
                    new_rule = Rule(rule.LHS, RHS)
                    rules.append(new_rule)
                    rules_dict[new_rule.LHS].append(new_rule.RHS)

                if not new_rule:
                    yardstick.append(rule)

            if len(unit_rules) == len(yardstick):
                break

            unit_rules = yardstick

        for rule in unit_rules:
            warn('Dead-end unit rule discarded: %s' % rule, Warning, 2)

        self.rules = list(set(rules))

    def _long_productions(self):
        '''Eliminate non-binary non-terminals.'''
        def bin_rule(rule):
            if rule.is_long():
                long_rules.append(rule)

            else:
                rules.append(rule)

        rules = []
        long_rules = []
        new_rules = {}
        n = 1

        for rule in self.rules:
            bin_rule(rule)

        while long_rules:
            rule = long_rules.pop()
            leftmost = rule.RHS[0:2]

            try:
                sym = new_rules[str(leftmost)]

            except KeyError:
                sym = 'X%i' % n  # NOTE
                n += 1
                rules.append(Rule(sym, leftmost))
                new_rules[str(leftmost)] = sym

            bin_rule(Rule(rule.LHS, [sym] + rule.RHS[2:]))

        self.rules = rules

    def _clean(self):
        '''Clean up and create disjunctive lexical rules.'''
        rules = []
        terminals = defaultdict(list)

        for rule in self.rules:
            if rule.is_terminal():
                terminals[rule.LHS].extend(rule.RHS)

            else:
                rules.append(rule)

        for LHS, RHS in terminals.items():
            rules.append(Rule(LHS, [' | '.join(RHS)]))

        rules.sort(key=lambda r: (r.LHS != self.start, r.LHS, r.is_terminal()))

        self.rules = rules


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_grammar', '-i')
    args = parser.parse_args()

    CNF(args.input_grammar)
