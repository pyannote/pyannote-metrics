#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS

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

"""
Evaluation metrics

Usage:
  eval identification [options] <references.mdtm> <hypothesis.mdtm>
  eval -h | --help
  eval --version

Options:
  --uem <eval.uem>         Evaluation map.
  --uris <uris.lst>        List of resources.
  -h --help                Show this screen.
  --version                Show version.
"""

import pyannote.metrics
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.parser.util import CoParser
from docopt import docopt
import sys


def do_identification(
    references_mdtm, hypothesis_mdtm, uris_lst=None, eval_uem=None
):

    ier = IdentificationErrorRate()

    iter_over = {
        'reference': references_mdtm,
        'hypothesis': hypothesis_mdtm
    }

    if uris_lst:
        iter_over['uris'] = uris_lst
    else:
        iter_over['uris'] = 'reference'

    if eval_uem:
        iter_over['uem'] = eval_uem

    coParser = CoParser(**iter_over)

    for uri, ref, hyp, uem in coParser.iter('uris', 'reference', 'hypothesis',
                                            'uem'):
        rate = ier(ref, hyp, uem=uem)
        sys.stdout.write('{uri:s}: {ier:3.2f}%\n'.format(uri=uri, ier=100 * rate))
        sys.stdout.flush()

    sys.stdout.write('Total: {ier:3.2f}%'.format(ier=100 * abs(ier)))

if __name__ == '__main__':

    version = 'PyAnnoteEval v{version:s}'.format(
        version=pyannote.metrics.__version__)
    arguments = docopt(__doc__, version=version)

    if arguments['identification']:
        references_mdtm = arguments['<references.mdtm>']
        hypothesis_mdtm = arguments['<hypothesis.mdtm>']
        uris_lst = arguments['--uris']
        eval_uem = arguments['--uem']

        do_identification(references_mdtm, hypothesis_mdtm,
                          uris_lst=uris_lst, eval_uem=eval_uem)
