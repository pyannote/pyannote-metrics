#!/usr/bin/env python
# encoding: utf-8

# Copyright 2012-2014 CNRS (Herve BREDIN -- bredin@limsi.fr)

# This file is part of PyAnnote.
#
#     PyAnnote is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     PyAnnote is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with PyAnnote.  If not, see <http://www.gnu.org/licenses/>.


from pyannote.core import Annotation, Timeline


class SegmentationError(object):

    def __init__(self):
        super(SegmentationError, self).__init__()

    def __call__(self, reference, hypothesis):

        if isinstance(reference, Annotation):
            reference = reference.get_timeline()

        if isinstance(hypothesis, Annotation):
            hypothesis = hypothesis.get_timeline()

        # over-segmentation
        over = Timeline(uri=reference.uri)
        prev_r = reference[0]
        intersection = []
        for r, h in reference.co_iter(hypothesis):

            if r != prev_r:
                intersection = sorted(intersection)
                for _, segment in intersection[:-1]:
                    over.add(segment)
                intersection = []
                prev_r = r

            segment = r & h
            intersection.append((segment.duration, segment))

        intersection = sorted(intersection)
        for _, segment in intersection[:-1]:
            over.add(segment)

        # under-segmentation
        under = Timeline(uri=reference.uri)
        prev_h = hypothesis[0]
        intersection = []
        for h, r in hypothesis.co_iter(reference):

            if h != prev_h:
                intersection = sorted(intersection)
                for _, segment in intersection[:-1]:
                    under.add(segment)
                intersection = []
                prev_h = h

            segment = h & r
            intersection.append((segment.duration, segment))

        intersection = sorted(intersection)
        for _, segment in intersection[:-1]:
            under.add(segment)

        # extent
        extent = reference.extent()

        # correct (neither under- nor over-segmented)
        correct = under.union(over).gaps(focus=extent)

        # frontier error (both under- and over-segmented)
        frontier = under.crop(over)

        # under-segmented
        not_over = over.gaps(focus=extent)
        only_under = under.crop(not_over)

        # over-segmented
        not_under = under.gaps(focus=extent)
        only_over = over.crop(not_under)

        status = Annotation(uri=reference.uri)
        for segment in correct:
            status[segment, '_'] = 'correct'
        for segment in frontier:
            status[segment, '_'] = 'frontier'
        for segment in only_over:
            status[segment, '_'] = 'over'
        for segment in only_under:
            status[segment, '_'] = 'under'

        return status.smooth()


