# pyannote.metrics

> a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems

An overview of `pyannote.metrics` is available as a [technical report](doc/pyannote-metrics.pdf): it is recommended to read it first, to quickly get an idea whether this tool is for you.

## Installation

```bash
$ conda create -n pyannote-metrics python=3.5 anaconda
$ source activate pyannote-metrics
$ conda install gcc
$ pip install pyannote.metrics
```

## Documentation

The documentation is not yet on par with that of [pyannote.core](http://pyannote.github.io/pyannote-core). We are working on it...

### Application programming interface (API)

[See example notebooks](http://nbviewer.ipython.org/github/pyannote/pyannote-metrics/blob/master/doc/index.ipynb)

### Command line interface (CLI)

```bash
$ pyannote.metrics.py --help
```

## Citation

If you use `pyannote.metrics` in your research, please use the following citation:

```bibtex
@techreport{pyannote.metrics,
  author = {Herv\'e Bredin},
  title = {{pyannote.metrics: a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems}},
  url = {http://pyannote.github.io},
}
```
