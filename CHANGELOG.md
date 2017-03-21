### Version 0.14.2 (2017-03-21)

  - feat: better README and technical report

### Version 0.14.1 (2017-03-16)

  - chore: rename SegmentationError to SegmentationErrorAnalysis
  - fix: fix DetectionErrorRate support for kwargs

### Version 0.14 (2017-02-06)

  - feat: add "parallel" option to not use multiprocessing
  - feat: add "accuracy" in "detection" report
  - setup: switch to pyannote.core 0.13
  - setup: switch to pyannote.parser 0.6.5

### Version 0.13.2 (2017-01-30)

  - feat: add pyannote-metrics.py evaluation script
  - fix: fix BaseMetric.report() for metric without a 'total' component
  - fix: fix (Greedy)DiarizationErrorRate uem handling
  - fix: fix (Greedy)DiarizationErrorRate parallel processing
  - setup: switch to pyannote.core 0.12
  - setup: update munkres & matplotlib dependencies

### Version 0.12.1 (2017-01-27)

  - feat: support for multiprocessing
  - feat: add report() method
  - feat: travis continuous integration (finally!)
  - improve: speed up detection metrics
  - feat: add unit tests for detection metrics
  - fix: fix python 3 support
  - setup: remove dependency to pyannote.algorithms
  - setup: switch to pyannote.core 0.11

### Version 0.11 (2016-12-13)

  - feat: add pyannote.metrics.binary_classification module

### Version 0.10.3 (2016-11-28)

  - fix: fix (greedy) diarization error rate
  - feat: add support for 'collar' to (greedy) diarization error rate

### Version 0.10.2 (2016-11-10)

  - fix: fix default "xlim" in "plot_distributions"
  - setup: switch to pyannote.core 0.8 and pyannote.algorithms 0.6.6

### Version 0.10.1 (2016-11-05)

  - feat: add "uem" support to diarization metrics

### Version 0.9 (2016-09-23)

  - feat: add plotting functions for binary classification tasks

### Version 0.8 (2016-08-25)

  - feat: detection accuracy
  - refactor: detection metrics
  - setup: update to pyannote.core 0.7.2

### Version 0.7.1 (2016-06-24)

  - setup: update to pyannote.core 0.6.6

### Version 0.7 (2016-04-04)

  - feat: greedy diarization error rate

### Version 0.6.0 (2016-03-29)

  - feat: Python 3 support
  - feat: unit tests
  - wip: travis

### Version 0.5.1 (2016-02-19)

  - refactor: diarization metrics

### Version 0.4.1 (2014-11-20)

  - fix: identification error analysis matrix confusion

### Version 0.4 (2014-10-31)

  - feat(error): identification regression analysis
  - feat: new pyannote_eval.py CLI

### Version 0.3 (2014-10-01)

  - feat(error): segmentation error analysis

### Version 0.2 (2014-08-05)

  - feat(detection): add precision and recall
  - fix(identification): fix precision and recall

### Version 0.1 (2014-06-27)

  - feat(segmentation): add precision and recall
  - feat(identification): add support for NIST collar
  - feat(error): add module for detailed error analysis

### Version 0.0.1 (2014-06-04)

  - first public version
