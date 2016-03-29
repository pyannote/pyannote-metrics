#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2016 CNRS

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

from pyannote.core import Timeline, Segment


class UEMSupportMixin:
    """Provides 'uemify' method with optional (à la NIST) collar"""

    def _get_collar(self, reference, duration):

        # initialize empty timeline
        collar = Timeline(uri=reference.uri)

        if duration == 0.:
            return collar

        # iterate over all segments in reference
        for segment in reference.itersegments():

            # add collar centered on start time
            t = segment.start
            collar.add(Segment(t - .5 * duration, t + .5 * duration))

            # add collar centered on end time
            t = segment.end
            collar.add(Segment(t - .5 * duration, t + .5 * duration))

        # merge overlapping collars and return
        return collar.coverage()

    def uemify(self, reference, hypothesis, uem=None, collar=0.):
        """

        Parameters
        ----------
        reference, hypothesis : Annotation
            Reference and hypothesis annotations.
        uem : Timeline, optional
            Evaluation map.
        collar : float, optional
            Duration (in seconds) of collars removed from evaluation around
            boundaries of reference segments.
        """

        # when uem is not provided
        # use the union of reference and hypothesis extents
        if uem is None:
            r_extent = reference.get_timeline().extent()
            h_extent = hypothesis.get_timeline().extent()
            uem = Timeline(segments=[r_extent | h_extent], uri=reference.uri)

        # remove collars from uem
        uem = self._get_collar(reference, collar).gaps(focus=uem)

        # crop reference and hypothesis
        reference = reference.crop(uem, mode='intersection')
        hypothesis = hypothesis.crop(uem, mode='intersection')

        return reference, hypothesis
