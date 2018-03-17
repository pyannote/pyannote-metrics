#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2018 CNRS

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
  pyannote-metrics.py detection [--subset=<subset> --collar=<seconds> --skip-overlap] <database.task.protocol> <hypothesis.mdtm>
  pyannote-metrics.py segmentation [--subset=<subset> --tolerance=<seconds>] <database.task.protocol> <hypothesis.mdtm>
  pyannote-metrics.py diarization [--subset=<subset> --greedy --collar=<seconds> --skip-overlap] <database.task.protocol> <hypothesis.mdtm>
  pyannote-metrics.py identification [--subset=<subset> --collar=<seconds> --skip-overlap] <database.task.protocol> <hypothesis.mdtm>
  pyannote-metrics.py spotting [--subset=<subset> --latency=<seconds>... --filter=<expression>...] <database.task.protocol> <hypothesis.json>
  pyannote-metrics.py -h | --help
  pyannote-metrics.py --version

Options:
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  --subset=<subset>          Evaluated subset (train|developement|test) [default: test]
  --collar=<seconds>         Collar, in seconds [default: 0.0].
  --skip-overlap             Do not evaluate overlap regions.
  --tolerance=<seconds>      Tolerance, in seconds [default: 0.5].
  --greedy                   Use greedy diarization error rate.
  --latency=<seconds>        Evaluate with fixed latency.
  --filter=<expression>      Filter out target trials that do not match the
                             expression; e.g. use --filter="speech>10" to skip
                             target trials with less than 10s of speech from
                             the target.
  -h --help                  Show this screen.
  --version                  Show version.

All modes but "spotting" expect hypothesis using the MDTM file format.
MDTM files contain one line per speech turn, using the following convention:

<uri> 1 <start_time> <duration> speaker <confidence> <gender> <speaker_id>

    * uri: file identifier (as given by pyannote.database protocols)
    * start_time: speech turn start time in seconds
    * duration: speech turn duration in seconds
    * confidence: confidence score (can be anything, not used for now)
    * gender: speaker gender (can be anything, not used for now)
    * speaker_id: speaker identifier

"spotting" mode expects hypothesis using the following JSON file format.
It should contain a list of trial hypothesis, using the same trial order as
pyannote.database speaker spotting protocols (e.g. protocol.test_trial())

[
    {'uri': '<uri>', 'model_id': '<model_id>', 'scores': [[<t1>, <v1>], [<t2>, <v2>], ... [<tn>, <vn>]]},
    {'uri': '<uri>', 'model_id': '<model_id>', 'scores': [[<t1>, <v1>], [<t2>, <v2>], ... [<tn>, <vn>]]},
    {'uri': '<uri>', 'model_id': '<model_id>', 'scores': [[<t1>, <v1>], [<t2>, <v2>], ... [<tn>, <vn>]]},
    ...
    {'uri': '<uri>', 'model_id': '<model_id>', 'scores': [[<t1>, <v1>], [<t2>, <v2>], ... [<tn>, <vn>]]},
]

    * uri: file identifier (as given by pyannote.database protocols)
    * model_id: target identifier (as given by pyannote.database protocols)
    * [ti, vi]: [time, value] pair indicating that the system has output the
                score vi at time ti (e.g. [10.2, 0.2] means that the system
                gave a score of 0.2 at time 10.2s).

Calling "spotting" mode will create a bunch of files.
* <hypothesis.det.txt> contains DET curve using the following raw file format:
    <threshold> <fpr> <fnr>
* <hypothesis.lcy.txt> contains latency curves using this format:
    <threshold> <fpr> <fnr> <speaker_latency> <absolute_latency>

"""


# command line parsing
from docopt import docopt

import sys
import json
import warnings
import functools
import numpy as np
import pandas as pd
from tabulate import tabulate
# import multiprocessing as mp

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

from pyannote.metrics.spotting import LowLatencySpeakerSpotting

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

    for item in items:
        process(item)

    # HB. 2018-02-05: parallel processing was removed because it is not clear
    # how to handle the case where the same 'uri' is processed several times
    # in a possibly different order for each sub-metric...
    # # heuristic to estimate the optimal number of processes
    # chunksize = 20
    # processes = max(1, min(mp.cpu_count(), n_items // chunksize))
    # pool = mp.Pool(processes)
    # _ = pool.map(process, items, chunksize=chunksize)

    return {key: metric.report(display=False)
            for key, metric in metrics.items()}

def reindex(report):
    """Reindex report so that 'TOTAL' is the last row"""
    index = list(report.index)
    i = index.index('TOTAL')
    return report.reindex(index[:i] + index[i+1:] + ['TOTAL'])

def detection(protocol, subset, hypotheses, collar=0.0, skip_overlap=False):

    options = {'collar': collar,
               'skip_overlap': skip_overlap,
               'parallel': True}

    metrics = {
        'error': DetectionErrorRate(**options),
        'accuracy': DetectionAccuracy(**options),
        'precision': DetectionPrecision(**options),
        'recall': DetectionRecall(**options)}

    reports = get_reports(protocol, subset, hypotheses, metrics)

    report = metrics['error'].report(display=False)
    accuracy = metrics['accuracy'].report(display=False)
    precision = metrics['precision'].report(display=False)
    recall = metrics['recall'].report(display=False)

    report['accuracy', '%'] = accuracy[metrics['accuracy'].name, '%']
    report['precision', '%'] = precision[metrics['precision'].name, '%']
    report['recall', '%'] = recall[metrics['recall'].name, '%']

    report = reindex(report)

    columns = list(report.columns)
    report = report[[columns[0]] + columns[-3:] + columns[1:-3]]

    summary = 'Detection (collar = {0:g} ms{1})'.format(
        1000*collar, ', no overlap' if skip_overlap else '')

    headers = [summary] + \
              [report.columns[i][0] for i in range(4)] + \
              ['%' if c[1] == '%' else c[0] for c in report.columns[4:]]

    print(tabulate(report, headers=headers, tablefmt="simple",
                   floatfmt=".2f", numalign="decimal", stralign="left",
                   missingval="", showindex="default", disable_numparse=False))

def segmentation(protocol, subset, hypotheses, tolerance=0.5):

    options = {'tolerance': tolerance, 'parallel': True}

    metrics = {'coverage': SegmentationCoverage(**options),
               'purity': SegmentationPurity(**options),
               'precision': SegmentationPrecision(**options),
               'recall': SegmentationRecall(**options)}

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
    report = reindex(report)

    headers = ['Segmentation (tolerance = {0:g} ms)'.format(1000*tolerance),
               'coverage', 'purity', 'precision', 'recall']
    print(tabulate(report, headers=headers, tablefmt="simple",
                   floatfmt=".2f", numalign="decimal", stralign="left",
                   missingval="", showindex="default", disable_numparse=False))

def diarization(protocol, subset, hypotheses, greedy=False,
                collar=0.0, skip_overlap=False):

    options = {'collar': collar,
               'skip_overlap': skip_overlap,
               'parallel': True}

    metrics = {
        'purity': DiarizationPurity(**options),
        'coverage': DiarizationCoverage(**options)}

    if greedy:
        metrics['error'] = GreedyDiarizationErrorRate(**options)
    else:
        metrics['error'] = DiarizationErrorRate(**options)

    reports = get_reports(protocol, subset, hypotheses, metrics)

    report = metrics['error'].report(display=False)
    purity = metrics['purity'].report(display=False)
    coverage = metrics['coverage'].report(display=False)

    report['purity', '%'] = purity[metrics['purity'].name, '%']
    report['coverage', '%'] = coverage[metrics['coverage'].name, '%']

    columns = list(report.columns)
    report = report[[columns[0]] + columns[-2:] + columns[1:-2]]

    report = reindex(report)

    summary = 'Diarization ({0:s}collar = {1:g} ms{2})'.format(
                'greedy, ' if greedy else '',
                1000 * collar,
                ', no overlap' if skip_overlap else '')

    headers = [summary] + \
              [report.columns[i][0] for i in range(3)] + \
              ['%' if c[1] == '%' else c[0] for c in report.columns[3:]]

    print(tabulate(report, headers=headers, tablefmt="simple",
                   floatfmt=".2f", numalign="decimal", stralign="left",
                   missingval="", showindex="default", disable_numparse=False))

def identification(protocol, subset, hypotheses,
                   collar=0.0, skip_overlap=False):

    options = {'collar': collar,
               'skip_overlap': skip_overlap,
               'parallel': True}

    metrics = {
        'error': IdentificationErrorRate(**options),
        'precision': IdentificationPrecision(**options),
        'recall': IdentificationRecall(**options)}

    reports = get_reports(protocol, subset, hypotheses, metrics)

    report = metrics['error'].report(display=False)
    precision = metrics['precision'].report(display=False)
    recall = metrics['recall'].report(display=False)

    report['precision', '%'] = precision[metrics['precision'].name, '%']
    report['recall', '%'] = recall[metrics['recall'].name, '%']

    columns = list(report.columns)
    report = report[[columns[0]] + columns[-2:] + columns[1:-2]]

    report = reindex(report)

    summary = 'Identification (collar = {1:g} ms{2})'.format(
                1000 * collar,
                ', no overlap' if skip_overlap else '')

    headers = [summary] + \
              [report.columns[i][0] for i in range(3)] + \
              ['%' if c[1] == '%' else c[0] for c in report.columns[3:]]

    print(tabulate(report, headers=headers, tablefmt="simple",
                   floatfmt=".2f", numalign="decimal", stralign="left",
                   missingval="", showindex="default", disable_numparse=False))

def spotting(protocol, subset, latencies, hypotheses, output_prefix,
             filter_func=None):

    if not latencies:
        Scores = []

    protocol.diarization = False

    trials = getattr(protocol, '{subset}_trial'.format(subset=subset))()
    for i, (current_trial, hypothesis) in enumerate(zip(trials, hypotheses)):

        # check trial/hypothesis target consistency
        try:
            assert current_trial['model_id'] == hypothesis['model_id']
        except AssertionError as e:
            msg = ('target mismatch in trial #{i} '
                   '(found: {found}, should be: {should_be})')
            raise ValueError(
                msg.format(i=i, found=hypothesis['model_id'],
                           should_be=current_trial['model_id']))

        # check trial/hypothesis file consistency
        try:
            assert current_trial['uri'] == hypothesis['uri']
        except AssertionError as e:
            msg = ('file mismatch in trial #{i} '
                   '(found: {found}, should be: {should_be})')
            raise ValueError(
                msg.format(i=i, found=hypothesis['uri'],
                           should_be=current_trial['uri']))

        # check at least one score is provided
        try:
            assert len(hypothesis['scores']) > 0
        except AssertionError as e:
            msg = ('empty list of scores in trial #{i}.')
            raise ValueError(msg.format(i=i))

        timestamps, scores = zip(*hypothesis['scores'])

        if not latencies:
            Scores.append(scores)

        # check trial/hypothesis timerange consistency
        try_with = current_trial['try_with']
        try:
            assert min(timestamps) >= try_with.start
        except AssertionError as e:
            msg = ('incorrect timestamp in trial #{i} '
                   '(found: {found:g}, should be: >= {should_be:g})')
            raise ValueError(
                msg.format(i=i,
                found=min(timestamps),
                should_be=try_with.start))

    if not latencies:
        # estimate best set of thresholds
        scores = np.concatenate(Scores)
        epsilons = np.array(
            [n * 10**(-e) for e in range(4, 1, -1) for n in range(1, 10)])
        percentile = np.concatenate([epsilons, np.arange(0.1, 100., 0.1), 100 - epsilons[::-1]])
        thresholds = np.percentile(scores, percentile)

    if not latencies:
        metric = LowLatencySpeakerSpotting(thresholds=thresholds)

    else:
        metric = LowLatencySpeakerSpotting(latencies=latencies)

    trials = getattr(protocol, '{subset}_trial'.format(subset=subset))()
    for i, (current_trial, hypothesis) in enumerate(zip(trials, hypotheses)):

        if filter_func is not None:
            speech = current_trial['reference'].duration()
            target_trial = speech > 0
            if target_trial and filter_func(speech):
                continue

        reference = current_trial['reference']
        metric(reference, hypothesis['scores'])

    if not latencies:

        thresholds, fpr, fnr, eer, _ = metric.det_curve(return_latency=False)

        # save DET curve to hypothesis.det.txt
        det_path = '{output_prefix}.det.txt'.format(output_prefix=output_prefix)
        det_tmpl = '{t:.9f} {p:.9f} {n:.9f}\n'
        with open(det_path, mode='w') as fp:
            fp.write('# threshold false_positive_rate false_negative_rate\n')
            for t, p, n in zip(thresholds, fpr, fnr):
                line = det_tmpl.format(t=t, p=p, n=n)
                fp.write(line)

        print('> {det_path}'.format(det_path=det_path))

        thresholds, fpr, fnr, _, _, speaker_lcy, absolute_lcy = \
            metric.det_curve(return_latency=True)

        # save DET curve to hypothesis.det.txt
        lcy_path = '{output_prefix}.lcy.txt'.format(output_prefix=output_prefix)
        lcy_tmpl = '{t:.9f} {p:.9f} {n:.9f} {s:.6f} {a:.6f}\n'
        with open(lcy_path, mode='w') as fp:
            fp.write('# threshold false_positive_rate false_negative_rate speaker_latency absolute_latency\n')
            for t, p, n, s, a in zip(thresholds, fpr, fnr, speaker_lcy, absolute_lcy):
                if p == 1:
                    continue
                if np.isnan(s):
                    continue
                line = lcy_tmpl.format(t=t, p=p, n=n, s=s, a=a)
                fp.write(line)

        print('> {lcy_path}'.format(lcy_path=lcy_path))

        print()
        print('EER% = {eer:.2f}'.format(eer=100 * eer))

    else:

        results = metric.det_curve()
        logs = []
        for key in sorted(results):

            result = results[key]
            log = {'latency': key}
            for latency in latencies:
                thresholds, fpr, fnr, eer, _ = result[latency]
                #print('EER @ {latency}s = {eer:.2f}%'.format(latency=latency,
                #                                             eer=100 * eer))
                log[latency] = eer
                # save DET curve to hypothesis.det.{lcy}s.txt
                det_path = '{output_prefix}.det.{key}.{latency:g}s.txt'.format(
                    output_prefix=output_prefix, key=key, latency=latency)
                det_tmpl = '{t:.9f} {p:.9f} {n:.9f}\n'
                with open(det_path, mode='w') as fp:
                    fp.write('# threshold false_positive_rate false_negative_rate\n')
                    for t, p, n in zip(thresholds, fpr, fnr):
                        line = det_tmpl.format(t=t, p=p, n=n)
                        fp.write(line)
            logs.append(log)
            det_path = '{output_prefix}.det.{key}.XXs.txt'.format(
                output_prefix=output_prefix, key=key)
            print('> {det_path}'.format(det_path=det_path))

        print()
        df = 100 * pd.DataFrame.from_dict(logs).set_index('latency')[latencies]
        print(tabulate(df, tablefmt="simple",
                       headers=['latency'] + ['EER% @ {l:g}s'.format(l=l) for l in latencies],
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
    skip_overlap = arguments['--skip-overlap']
    tolerance = float(arguments['--tolerance'])

    if arguments['spotting']:

        hypothesis_json = arguments['<hypothesis.json>']
        with open(hypothesis_json, mode='r') as fp:
            hypotheses = json.load(fp)

        output_prefix = hypothesis_json[:-5]

        latencies = [float(l) for l in arguments['--latency']]

        filters = arguments['--filter']
        if filters:
            from sympy import sympify, lambdify, symbols
            speech = symbols('speech')
            filter_funcs = []
            filter_funcs = [
                lambdify([speech], sympify(expression))
                for expression in filters]
            filter_func = lambda speech: \
                any(~func(speech) for func in filter_funcs)
        else:
            filter_func = None

        spotting(protocol, subset, latencies, hypotheses, output_prefix,
                 filter_func=filter_func)

        sys.exit(0)

    # hypothesis
    hypothesis_mdtm = arguments['<hypothesis.mdtm>']
    hypotheses = MagicParser().read(hypothesis_mdtm, modality='speaker')

    if arguments['detection']:
        detection(protocol, subset, hypotheses,
                  collar=collar, skip_overlap=skip_overlap)

    if arguments['segmentation']:
        segmentation(protocol, subset, hypotheses, tolerance=tolerance)

    if arguments['diarization']:
        greedy = arguments['--greedy']
        diarization(protocol, subset, hypotheses, greedy=greedy,
                    collar=collar, skip_overlap=skip_overlap)

    if arguments['identification']:
        identification(protocol, subset, hypotheses,
                       collar=collar, skip_overlap=skip_overlap)
