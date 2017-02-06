#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

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
Evaluation

Usage:
  evaluation detection [options] [--collar=<seconds>] <database.task.protocol> <hypothesis.mdtm>
  evaluation segmentation [options] [--tolerance=<seconds>] <database.task.protocol> <hypothesis.mdtm>
  evaluation diarization [options] [--greedy] [--collar=<seconds>] <database.task.protocol> <hypothesis.mdtm>
  evaluation identification [options] [--collar=<seconds>] <database.task.protocol> <hypothesis.mdtm>
  evaluation -h | --help
  evaluation --version

Options:
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  --subset=<subset>          Evaluated subset (train|developement|test) [default: test]
  --collar=<seconds>         Collar, in seconds [default: 0.0].
  --tolerance=<seconds>      Tolerance, in seconds [default: 0.5].
  --greedy                   Use greedy diarization error rate.
  -h --help                  Show this screen.
  --version                  Show version.
"""


# command line parsing
from docopt import docopt

import warnings
import functools
import pandas as pd
from tabulate import tabulate
import multiprocessing as mp

# use for parsing hypothesis file
from pyannote.parser import MagicParser

# evaluation protocols
from pyannote.database import get_protocol
from pyannote.database.util import get_annotated

from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.detection import DetectionAccuracy
from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionPrecision

from pyannote.metrics.segmentation import SegmentationPurity
from pyannote.metrics.segmentation import SegmentationCoverage
from pyannote.metrics.segmentation import SegmentationPrecision
from pyannote.metrics.segmentation import SegmentationRecall

from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.diarization import DiarizationPurity
from pyannote.metrics.diarization import DiarizationCoverage

from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.metrics.identification import IdentificationPrecision
from pyannote.metrics.identification import IdentificationRecall

showwarning_orig = warnings.showwarning

def showwarning(message, category, *args, **kwargs):
    import sys
    print(category.__name__ + ':', str(message))

warnings.showwarning = showwarning

def get_hypothesis(hypotheses, item):

    uri = item['uri']

    if uri in hypotheses.uris:
        hypothesis = hypotheses(uri=uri)
    else:
        # if the exact 'uri' is not available in hypothesis,
        # look for matching substring
        tmp_uri = [u for u in hypotheses.uris if u in uri]
        if len(tmp_uri) == 0:
            msg = 'Could not find hypothesis for file "{uri}".'
            raise ValueError(msg.format(uri=uri))
        elif len(tmp_uri) > 1:
            msg = 'Found too many hypotheses matching file "{uri}" ({uris}).'
            raise ValueError(msg.format(uri=uri, uris=tmp_uri))
        else:
            tmp_uri = tmp_uri[0]
            msg = 'Could not find hypothesis for file "{uri}"; using "{tmp_uri}" instead.'
            warnings.warn(msg.format(tmp_uri=tmp_uri, uri=uri))

        hypothesis = hypotheses(uri=tmp_uri)
        hypothesis.uri = uri

    return hypothesis

def process_one(item, hypotheses=None, metrics=None):
    reference = item['annotation']
    hypothesis = get_hypothesis(hypotheses, item)
    uem = get_annotated(item)
    return {key: metric(reference, hypothesis, uem=uem)
            for key, metric in metrics.items()}

def get_reports(protocol, subset, hypotheses, metrics):

    process = functools.partial(process_one,
                                hypotheses=hypotheses,
                                metrics=metrics)

    # get items and their number
    progress = protocol.progress
    protocol.progress = False
    items = list(getattr(protocol, subset)())
    protocol.progress = progress
    n_items = len(items)

    # heuristic to estimate the optimal number of processes
    chunksize = 20
    processes = max(1, min(mp.cpu_count(), n_items // chunksize))

    pool = mp.Pool(processes)
    _ = pool.map(process, items, chunksize=chunksize)

    return {key: metric.report(display=False)
            for key, metric in metrics.items()}

def reindex(report, protocol, subset):
    progress = protocol.progress
    protocol.progress = False
    new_index = [item['uri'] for item in getattr(protocol, subset)()] + \
                ['TOTAL']
    protocol.progress = progress
    return report.reindex(new_index)

def detection(protocol, subset, hypotheses, collar=0.0):

    metrics = {'error': DetectionErrorRate(collar=collar),
               'accuracy': DetectionAccuracy(collar=collar),
               'precision': DetectionPrecision(collar=collar),
               'recall': DetectionRecall(collar=collar)}

    reports = get_reports(protocol, subset, hypotheses, metrics)

    report = metrics['error'].report(display=False)
    accuracy = metrics['accuracy'].report(display=False)
    precision = metrics['precision'].report(display=False)
    recall = metrics['recall'].report(display=False)

    report['accuracy', '%'] = accuracy[metrics['accuracy'].name, '%']
    report['precision', '%'] = precision[metrics['precision'].name, '%']
    report['recall', '%'] = recall[metrics['recall'].name, '%']

    report = reindex(report, protocol, subset)

    columns = list(report.columns)
    report = report[[columns[0]] + columns[-3:] + columns[1:-3]]

    headers = ['Detection (collar = {0:g} ms)'.format(1000*collar)] + \
              [report.columns[i][0] for i in range(4)] + \
              ['%' if c[1] == '%' else c[0] for c in report.columns[4:]]

    print(tabulate(report, headers=headers, tablefmt="simple",
                   floatfmt=".2f", numalign="decimal", stralign="left",
                   missingval="", showindex="default", disable_numparse=False))

def segmentation(protocol, subset, hypotheses, tolerance=0.5):

    metrics = {'coverage': SegmentationCoverage(tolerance=tolerance),
               'purity': SegmentationPurity(tolerance=tolerance),
               'precision': SegmentationPrecision(tolerance=tolerance),
               'recall': SegmentationRecall(tolerance=tolerance)}

    reports = get_reports(protocol, subset, hypotheses, metrics)

    coverage = metrics['coverage'].report(display=False)
    purity = metrics['purity'].report(display=False)
    precision = metrics['precision'].report(display=False)
    recall = metrics['recall'].report(display=False)

    coverage = coverage[metrics['coverage'].name]
    purity = purity[metrics['purity'].name]
    precision = precision[metrics['precision'].name]
    recall = recall[metrics['recall'].name]

    report = pd.concat([coverage, purity, precision, recall], axis=1)
    report = reindex(report, protocol, subset)

    headers = ['Segmentation (tolerance = {0:g} ms)'.format(1000*tolerance),
               'coverage', 'purity', 'precision', 'recall']
    print(tabulate(report, headers=headers, tablefmt="simple",
                   floatfmt=".2f", numalign="decimal", stralign="left",
                   missingval="", showindex="default", disable_numparse=False))

def diarization(protocol, subset, hypotheses, collar=0.0, greedy=False):

    metrics = {'purity': DiarizationPurity(collar=collar),
               'coverage': DiarizationCoverage(collar=collar)}

    if greedy:
        metrics['error'] = GreedyDiarizationErrorRate(collar=collar)
    else:
        metrics['error'] = DiarizationErrorRate(collar=collar)

    reports = get_reports(protocol, subset, hypotheses, metrics)

    report = metrics['error'].report(display=False)
    purity = metrics['purity'].report(display=False)
    coverage = metrics['coverage'].report(display=False)

    report['purity', '%'] = purity[metrics['purity'].name, '%']
    report['coverage', '%'] = coverage[metrics['coverage'].name, '%']

    columns = list(report.columns)
    report = report[[columns[0]] + columns[-2:] + columns[1:-2]]

    report = reindex(report, protocol, subset)

    headers = ['Diarization ({0:s}collar = {1:g} ms)'.format(
                    'greedy, ' if greedy else '', 1000*collar)] + \
              [report.columns[i][0] for i in range(3)] + \
              ['%' if c[1] == '%' else c[0] for c in report.columns[3:]]

    print(tabulate(report, headers=headers, tablefmt="simple",
                   floatfmt=".2f", numalign="decimal", stralign="left",
                   missingval="", showindex="default", disable_numparse=False))


def identification(protocol, subset, hypotheses, collar=0.0):

    metrics = {'error': IdentificationErrorRate(collar=collar),
               'precision': IdentificationPrecision(collar=collar),
               'recall': IdentificationRecall(collar=collar)}

    reports = get_reports(protocol, subset, hypotheses, metrics)

    report = metrics['error'].report(display=False)
    precision = metrics['precision'].report(display=False)
    recall = metrics['recall'].report(display=False)

    report['precision', '%'] = precision[metrics['precision'].name, '%']
    report['recall', '%'] = recall[metrics['recall'].name, '%']

    columns = list(report.columns)
    report = report[[columns[0]] + columns[-2:] + columns[1:-2]]

    report = reindex(report, protocol, subset)

    headers = ['Identification (collar = {0:g} ms)'.format(1000*collar)] + \
              [report.columns[i][0] for i in range(3)] + \
              ['%' if c[1] == '%' else c[0] for c in report.columns[3:]]

    print(tabulate(report, headers=headers, tablefmt="simple",
                   floatfmt=".2f", numalign="decimal", stralign="left",
                   missingval="", showindex="default", disable_numparse=False))


if __name__ == '__main__':

    arguments = docopt(__doc__, version='Evaluation')

    # protocol
    protocol_name = arguments['<database.task.protocol>']
    protocol = get_protocol(protocol_name, progress=True)

    # subset (train, development, or test)
    subset = arguments['--subset']

    collar = float(arguments['--collar'])
    tolerance = float(arguments['--tolerance'])

    # hypothesis
    hypothesis_mdtm = arguments['<hypothesis.mdtm>']
    hypotheses = MagicParser().read(hypothesis_mdtm, modality='speaker')

    if arguments['detection']:
        detection(protocol, subset, hypotheses, collar=collar)

    if arguments['segmentation']:
        segmentation(protocol, subset, hypotheses, tolerance=tolerance)

    if arguments['diarization']:
        greedy = arguments['--greedy']
        diarization(protocol, subset, hypotheses, collar=collar, greedy=greedy)

    if arguments['identification']:
        identification(protocol, subset, hypotheses, collar=collar)
