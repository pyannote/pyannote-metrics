#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

from __future__ import unicode_literals

import numpy as np
from munkres import Munkres
import networkx as nx

MATCH_CORRECT = 'correct'
MATCH_CONFUSION = 'confusion'
MATCH_MISSED_DETECTION = 'missed detection'
MATCH_FALSE_ALARM = 'false alarm'
MATCH_TOTAL = 'total'


class LabelMatcher(object):
    """
    ID matcher base class.

    All ID matcher classes must inherit from this class and implement
    .match() -- ie return True if two IDs match and False
    otherwise.
    """

    def __init__(self):
        super(LabelMatcher, self).__init__()
        self._munkres = Munkres()

    def match(self, rlabel, hlabel):
        """
        Parameters
        ----------
        rlabel :
            Reference label
        hlabel :
            Hypothesis label

        Returns
        -------
        match : bool
            True if labels match, False otherwise.

        """
        # Two IDs match if they are equal to each other
        return rlabel == hlabel

    def __call__(self, rlabels, hlabels):
        """

        Parameters
        ----------
        rlabels, hlabels : iterable
            Reference and hypothesis labels

        Returns
        -------
        counts : dict
        details : dict

        """

        # counts and details
        counts = {
            MATCH_CORRECT: 0,
            MATCH_CONFUSION: 0,
            MATCH_MISSED_DETECTION: 0,
            MATCH_FALSE_ALARM: 0,
            MATCH_TOTAL: 0
        }

        details = {
            MATCH_CORRECT: [],
            MATCH_CONFUSION: [],
            MATCH_MISSED_DETECTION: [],
            MATCH_FALSE_ALARM: []
        }

        NR = len(rlabels)
        NH = len(hlabels)
        N = max(NR, NH)

        # corner case
        if N == 0:
            return (counts, details)

        # this is to make sure rlables and hlabels are lists
        # as we will access them later by index
        rlabels = list(rlabels)
        hlabels = list(hlabels)

        # initialize match matrix
        # with True if labels match and False otherwise
        match = np.zeros((N, N), dtype=bool)
        for r, rlabel in enumerate(rlabels):
            for h, hlabel in enumerate(hlabels):
                match[r, h] = self.match(rlabel, hlabel)

        # find one-to-one mapping that maximize total number of matches
        # using the Hungarian algorithm
        mapping = self._munkres.compute(1 - match)

        # loop on matches
        for r, h in mapping:

            # hypothesis label is matched with unexisting reference label
            # ==> this is a false alarm
            if r >= NR:
                counts[MATCH_FALSE_ALARM] += 1
                details[MATCH_FALSE_ALARM].append(hlabels[h])

            # reference label is matched with unexisting hypothesis label
            # ==> this is a missed detection
            elif h >= NH:
                counts[MATCH_MISSED_DETECTION] += 1
                details[MATCH_MISSED_DETECTION].append(rlabels[r])

            # reference and hypothesis labels match
            # ==> this is a correct detection
            elif match[r, h]:
                counts[MATCH_CORRECT] += 1
                details[MATCH_CORRECT].append((rlabels[r], hlabels[h]))

            # refernece and hypothesis do not match
            # ==> this is a confusion
            else:
                counts[MATCH_CONFUSION] += 1
                details[MATCH_CONFUSION].append((rlabels[r], hlabels[h]))

        counts[MATCH_TOTAL] += NR

        # returns counts and details
        return (counts, details)


class HungarianMapper(object):

    def __init__(self):
        super(HungarianMapper, self).__init__()
        self._munkres = Munkres()

    def _helper(self, A, B):

        # transpose matrix in case A has more labels than B
        Na = len(A.labels())
        Nb = len(B.labels())
        if Na > Nb:
            return {a: b for (b, a) in self._helper(B, A).items()}

        matrix = A * B
        mapping = self._munkres.compute(matrix.max() - matrix)

        return dict(
            (matrix.coords['i'][i].item(), matrix.coords['j'][j].item())
            for i, j in mapping if matrix[i, j] > 0)

    def __call__(self, A, B):

        # build bi-partite cooccurrence graph
        # ------------------------------------

        # labels from A are linked with labels from B
        # if and only if the co-occur
        cooccurrence_graph = nx.Graph()

        # for a_label in A.labels():
        #     a = ('A', a_label)
        #     cooccurrence_graph.add_node(a)
        #
        # for b_label in B.labels():
        #     b = ('B', b_label)
        #     cooccurrence_graph.add_node(b)

        for a_track, b_track in A.co_iter(B):
            a = ('A', A[a_track])
            b = ('B', B[b_track])
            cooccurrence_graph.add_edge(a, b)

        # divide & conquer
        # ------------------

        # split a (potentially large) association problem into smaller ones

        mapping = dict()

        for component in nx.connected_components(cooccurrence_graph):

            # extract smaller problems
            a_labels = [label for (src, label) in component if src == 'A']
            b_labels = [label for (src, label) in component if src == 'B']
            sub_A = A.subset(a_labels)
            sub_B = B.subset(b_labels)

            local_mapping = self._helper(sub_A, sub_B)
            mapping.update(local_mapping)

        return mapping


class GreedyMapper(object):

    def __call__(self, A, B):

        matrix = A * B
        Na, Nb = matrix.shape
        N = min(Na, Nb)

        mapping = {}

        for i in range(N):

            ab = np.argmax(matrix.data)
            a = ab // (Nb-i)
            b = ab % (Nb-i)

            cost = matrix[a, b].item()

            if cost == 0:
                break

            alabel = matrix.coords['i'][a].item()
            blabel = matrix.coords['j'][b].item()

            mapping[alabel] = blabel

            matrix = matrix.drop([alabel], dim='i').drop([blabel], dim='j')

        return mapping
