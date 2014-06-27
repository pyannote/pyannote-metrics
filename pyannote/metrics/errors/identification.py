#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2014 CNRS (Hervé BREDIN - http://herve.niderb.fr)

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

from __future__ import unicode_literals

import numpy as np

from ..matcher import LabelMatcherWithUnknownSupport
from pyannote.core import Annotation, Unknown
from pyannote.core.matrix import LabelMatrix
from pyannote.algorithms.tagging import DirectTagger

from ..matcher import MATCH_CORRECT, MATCH_CONFUSION, \
    MATCH_MISSED_DETECTION, MATCH_FALSE_ALARM

from ..identification import UEMSupportMixin

REFERENCE_TOTAL = 'reference'
HYPOTHESIS_TOTAL = 'hypothesis'


class IdentificationErrorAnalysis(UEMSupportMixin, object):
    """

    Parameters
    ----------
    matcher : `Matcher`, optional
        Defaults to `LabelMatcherWithUnknownSupport` instance
        i.e. two Unknowns are always considered as correct.
    unknown : bool, optional
        Set `unknown` to True (default) to take `Unknown` instances into
        account. Set it to False to get rid of them before evaluation.
    merge_unknowns : bool, optional
        See all `Unknown` instances as one unique label. Defaults to False.
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.

    """

    def __init__(self, matcher=None, unknown=True, merge_unknowns=False,
                 collar=0.):

        super(IdentificationErrorAnalysis, self).__init__()

        if matcher is None:
            matcher = LabelMatcherWithUnknownSupport()
        self.matcher = matcher
        self.unknown = unknown
        self.merge_unknowns = merge_unknowns
        self.collar = collar

        self._tagger = DirectTagger()

    def _merge_unknowns(self, reference, hypothesis):

        # create new unique `Unknown` instance label
        unknown = Unknown('?')

        # gather reference and hypothesis unknown labels
        rUnknown = [u for u in reference.labels()
                    if isinstance(u, Unknown)]
        hUnknown = [u for u in hypothesis.labels()
                    if isinstance(u, Unknown)]

        # replace them all by unique unknown label
        translation = {u: unknown for u in rUnknown + hUnknown}
        reference = reference.translate(translation)
        hypothesis = hypothesis.translate(translation)

        return reference, hypothesis

    def annotation(self, reference, hypothesis, uem=None, uemified=False):
        """Get error analysis as `Annotation`

        Labels are (status, reference_label, hypothesis_label) tuples.
        `status` is either 'correct', 'confusion', 'missed detection' or
        'false alarm'.
        `reference_label` is None in case of 'false alarm'.
        `hypothesis_label` is None in case of 'missed detection'.

        Parameters
        ----------
        uemified : bool, optional
            Returns "uemified" version of reference and hypothesis.
            Defaults to False.

        Returns
        -------
        errors : `Annotation`

        """

        reference, hypothesis = self.uemify(
            reference, hypothesis, uem=uem, collar=self.collar)

        # merge Unknown instance labels into one unique `Unknown` instance
        if self.unknown and self.merge_unknowns:
            reference, hypothesis = self._merge_unknowns(reference, hypothesis)

        # common (up-sampled) timeline
        common_timeline = reference.get_timeline().union(
            hypothesis.get_timeline())
        common_timeline = common_timeline.segmentation()

        # align reference on common timeline
        R = self._tagger(reference, common_timeline)

        # translate and align hypothesis on common timeline
        H = self._tagger(hypothesis, common_timeline)

        errors = Annotation(uri=reference.uri, modality=reference.modality)

        # loop on all segments
        for segment in common_timeline:

            # list of labels in reference segment
            rlabels = R.get_labels(segment, unknown=self.unknown, unique=False)

            # list of labels in hypothesis segment
            hlabels = H.get_labels(segment, unknown=self.unknown, unique=False)

            _, details = self.matcher(rlabels, hlabels)

            for r, h in details[MATCH_CORRECT]:
                track = errors.new_track(segment, prefix=MATCH_CORRECT)
                errors[segment, track] = (MATCH_CORRECT, r, h)

            for r, h in details[MATCH_CONFUSION]:
                track = errors.new_track(segment, prefix=MATCH_CONFUSION)
                errors[segment, track] = (MATCH_CONFUSION, r, h)

            for r in details[MATCH_MISSED_DETECTION]:
                track = errors.new_track(segment,
                                         prefix=MATCH_MISSED_DETECTION)
                errors[segment, track] = (MATCH_MISSED_DETECTION, r, None)

            for h in details[MATCH_FALSE_ALARM]:
                track = errors.new_track(segment, prefix=MATCH_FALSE_ALARM)
                errors[segment, track] = (MATCH_FALSE_ALARM, None, h)

        if uemified:
            return reference, hypothesis, errors
        else:
            return errors

    def matrix(self, reference, hypothesis, uem=None):

        reference, hypothesis, errors = self.annotation(
            reference, hypothesis, uem=uem, uemified=True)

        chart = errors.chart()

        # rLabels contains reference labels
        # hLabels contains hypothesis labels confused with a reference label
        # falseAlarmLabels contains false alarm hypothesis labels that do not
        # exist in reference labels // corner case  //
        rLabels, hLabels, falseAlarmLabels = set([]), set([]), set([])
        for (status, rLabel, hLabel), _ in chart:

            # labels for confusion matrix
            if status in {MATCH_CORRECT, MATCH_CONFUSION}:
                rLabels.add(rLabel)
                hLabels.add(hLabel)

            # missed reference label
            if status == MATCH_MISSED_DETECTION:
                rLabels.add(rLabel)

            # false alarm hypothesis labels
            if status == MATCH_FALSE_ALARM:
                falseAlarmLabels.add(hLabel)

        # make sure only labels that do not exist in reference are ketp
        falseAlarmLabels = falseAlarmLabels - rLabels

        # sort these sets of labels
        cmp_func = reference._cmp_labels
        falseAlarmLabels = sorted(falseAlarmLabels, cmp=cmp_func)
        rLabels = sorted(rLabels, cmp=cmp_func)
        hLabels = sorted(hLabels, cmp=cmp_func)

        # append false alarm labels as last 'reference' labels
        # (make sure to mark them as such)
        rLabels = rLabels + [(MATCH_FALSE_ALARM, hLabel)
                             for hLabel in falseAlarmLabels]

        # prepend duration columns before the detailed confusion matrix
        hLabels = [
            REFERENCE_TOTAL, HYPOTHESIS_TOTAL,
            MATCH_CORRECT, MATCH_CONFUSION,
            MATCH_FALSE_ALARM, MATCH_MISSED_DETECTION
        ] + hLabels

        # initialize empty matrix
        matrix = LabelMatrix(
            rows=rLabels, columns=hLabels,
            data=np.zeros((len(rLabels), len(hLabels)))
        )

        # loop on chart
        for (status, rLabel, hLabel), duration in chart:

            # increment confusion matrix
            if status in {MATCH_CORRECT, MATCH_CONFUSION}:
                matrix[rLabel, hLabel] += duration

            # corner case for ('false alarm', None, hLabel)
            if status == MATCH_FALSE_ALARM:
                # hLabel is also a reference label
                if hLabel in rLabels:
                    rLabel = hLabel
                # hLabel comes out of nowhere (ie. is not even in reference)
                else:
                    rLabel = (MATCH_FALSE_ALARM, hLabel)

            # increment status column
            matrix[rLabel, status] += duration

        # total reference and hypothesis duration
        for rLabel in rLabels:

            if isinstance(rLabel, tuple) and rLabel[0] == MATCH_FALSE_ALARM:
                r = 0.
                h = hypothesis.label_duration(rLabel[0])
            else:
                r = reference.label_duration(rLabel)
                h = hypothesis.label_duration(rLabel)

            matrix[rLabel, REFERENCE_TOTAL] = r
            matrix[rLabel, HYPOTHESIS_TOTAL] = h

        return matrix
