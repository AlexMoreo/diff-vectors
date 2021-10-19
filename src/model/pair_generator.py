import numpy as np
import itertools
import random

# create the samples for
# 1) training: positives (same author) and negatives (different authors)
# 2) test
# and transform them into vectors

# policies for the selection of the positive pairs

valid_positive_selection_policies = ['round_robin', 'random']


class PairGenerator:

    # function that controls the generation of the pairs (pos and neg)
    def __init__(self, target, pos_request, neg_request, max_request=-1, pos_selection_policy='round_robin',
                 force=False, verification_task=False):

        self.verification_task = verification_task

        if verification_task:
            assert set(np.unique(target)) == {0, 1}, 'verification task set with non-binary labels'

        assert pos_selection_policy in valid_positive_selection_policies, \
            f'unexpected pos_selection, valid ones are {valid_positive_selection_policies}'

        groups = self.get_groups(target)
        max_possible_positives = self.max_possible_positives(groups)
        max_possible_negatives = self.max_possible_negatives(groups)
        self.target = target
        # print(f'max number of documents reachable = {max_number}')
        self.pos_request = pos_request if pos_request != -1 else max_possible_positives
        self.neg_request = neg_request if neg_request != -1 else min(self.pos_request, max_possible_negatives)
        if max_request != -1:
            self.pos_request = min(self.pos_request, max_request)
            self.neg_request = min(self.neg_request, max_request)

        if force:
            raise NotImplementedError('force is not yet implemented')
        else:
            control_value = max_possible_positives
            assert pos_request <= control_value, \
                'the requested positive pairs exceed the number of combinations ({control_value})'
            control_value = max_possible_negatives
            assert neg_request <= control_value, \
                f'the requested negative pairs exceed the number of combinations ({control_value})'

        pairs = []
        # sampling of positive pairs (with selected policy)
        for pos_pair in self.positive_pair_generator(groups, pos_selection_policy):
            pairs.append(pos_pair)
            if len(pairs) >= self.pos_request:
                break
        positives = len(pairs)

        # sampling of negativenegative-taken pairs (assumed infinite)
        neg_register = set()
        for neg_pair in self.negative_pair_generator(groups):
            if neg_pair not in neg_register:
                neg_register.add(neg_pair)
                pairs.append(neg_pair)
                if len(pairs) - positives >= self.neg_request:
                    break
        negatives = len(pairs) - positives

        print(f'pairs generated={len(pairs)} (positive={positives}, negative={negatives})')

        self.pairs = np.asarray(pairs)
        self.labels = np.array([1] * positives + [0] * negatives)

    def ordered_pair(self, pair):
        a,b=pair
        return (a,b) if a <= b else (b,a)

    # function that create all the POSITIVE PAIRS, which are then taken depending on the pair_policy
    def positive_pair_generator(self, groups, pair_policy):
        if self.verification_task:
            positive_pairs = list(itertools.combinations(groups[1], 2))
            random.shuffle(positive_pairs)
            for p in positive_pairs:
                yield self.ordered_pair(p)
        else:
            positive_pairs = [list(itertools.combinations(group_i, 2)) for group_i in groups]
        positive_pairs = [p for p in positive_pairs if p]

        # policy where the pos pairs are taken one-from-one labelled group in sequence
        # (1 pair from group 1, 1 pair from group 2 etc)
        if pair_policy == 'round_robin':
            [random.shuffle(pos_pairs_i) for pos_pairs_i in positive_pairs if pos_pairs_i]
            while positive_pairs:
                pos_pairs_i = positive_pairs.pop(0)
                yield self.ordered_pair(pos_pairs_i.pop(0))
                if pos_pairs_i:
                    positive_pairs.append(pos_pairs_i)

        # policy where the pos pairs are taken one-from-one authorial group at random
        #(1 pair from group x, 1 pair from group y etc)
        elif pair_policy == 'random':
            positive_pairs = list(itertools.chain.from_iterable(positive_pairs))
            random.shuffle(positive_pairs)
            while positive_pairs:
                yield self.ordered_pair(positive_pairs.pop())

    # function that create the NEGATIVE PAIRS taking two docs randomly
    def negative_pair_generator(self, groups):
        all = sum([len(group_i) for group_i in groups])
        p = [len(group_i)/all for group_i in groups]
        while True:
            #group_a, group_b = np.random.choice(groups, size=2, replace=False, p=p)
            group_a, group_b = random.sample(list(groups), 2)
            pair=(np.random.choice(group_a), np.random.choice(group_b))
            yield self.ordered_pair(pair)

    # function that forms the labelled groups of docs
    def get_groups(self, target):
        indices = np.arange(len(target))
        groups = []
        for label in sorted(np.unique(target)):
            groups.append(indices[target == label])
        return groups

    # function that computes the maximum number of positive pairs that can be created
    def max_possible_positives(self, groups):
        if self.verification_task:
            lengths = [len(groups[1])]
        else:
            lengths = [len(group) for group in groups]
        control_value = sum([lengths_i*(lengths_i-1)/2 for lengths_i in lengths])
        return int(control_value)

    # function that computes the maximum number of negative pairs that can be created
    def max_possible_negatives(self, groups):
        # this function works well even if self.verification_task==True
        lengths = [len(group) for group in groups]
        control_value = 0
        for i in range(len(groups)-1):
            length_i = lengths[i]
            length_rest = sum(lengths[i+1:])
            control_value += length_i * length_rest
        return int(control_value)


