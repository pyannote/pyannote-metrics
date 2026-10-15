"""Smoke tests for the CLI, plotting, and transcription extras."""

import subprocess
import sys

import numpy as np
import pytest


def test_cli_help():
    pytest.importorskip("docopt")
    pytest.importorskip("tabulate")
    result = subprocess.run(
        [sys.executable, "-m", "pyannote.metrics.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Usage:" in result.stdout


def test_score_distribution_plot(tmp_path):
    pytest.importorskip("matplotlib")
    from pyannote.metrics.plot.binary_classification import plot_distributions

    target = tmp_path / "scores"
    plot_distributions(
        np.array([False, False, True, True]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        str(target),
    )
    assert target.with_suffix(".scores.png").stat().st_size > 0
    assert target.with_suffix(".scores.eps").stat().st_size > 0


def test_word_error_rate():
    pytest.importorskip("meeteval")
    from meeteval.io.seglst import SegLST

    from pyannote.metrics.transcription import WordErrorRate

    reference = SegLST(
        [{"session_id": "test", "start_time": 0, "end_time": 1, "words": "hello world"}]
    )
    hypothesis = SegLST(
        [{"session_id": "test", "start_time": 0, "end_time": 1, "words": "hello there"}]
    )
    assert WordErrorRate()(reference, hypothesis) == pytest.approx(0.5)


def test_missing_transcription_extra():
    code = """
import sys
sys.modules['meeteval'] = None
from pyannote.metrics.transcription import WordErrorRate
try:
    WordErrorRate()
except ImportError as error:
    assert 'transcription' in str(error)
else:
    raise AssertionError('Missing extra must produce an actionable ImportError')
"""
    subprocess.run([sys.executable, "-c", code], check=True)
