from pyannote.metrics.errors.identification import IdentificationErrorAnalysis
from pyannote.metrics.matcher import HungarianMapper
from collections import defaultdict
from category_stats import CategoryStats, DEFAULT_LABELS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import warnings
import os
import sys
import glob


"""
Generalised version of GenderErrorAnalysis. Now called CategoryErrorAnalysis so
it can work with any set of discrete annotation labels.

Default behaviour (labels=None) is identical to the original GenderErrorAnalysis:
  - label set  : ["MAL", "FEM", "OCH", "KCHI"]
  - child merge : OCH + KCHI → CHI  (in normalisation and confusion matrix)
  - confusion shorthand keys: "M", "F", "C"  (single-char, legacy behaviour)

With a custom label set:
  - no automatic merging
  - confusion matrix keys use full label strings
  - colours are auto-assigned unless a {label: hex} dict is passed
"""

# Palette used for auto-assigning colours when no colour dict is provided.
_AUTO_PALETTE = [
    '#008dff', '#ff9d3a', '#4ecb8d', '#e94f5b',
    '#9b5de5', '#f15bb5', '#fee440', '#00bbf9',
    '#0000A2', '#BC272D', '#E9C716', '#50AD9F',
]

_DEFAULT_COLORS = {'MAL': '#008dff', 'FEM': '#ff9d3a', 'CHI': '#4ecb8d'}


def _build_color_map(labels, user_colors=None):
    """
    Returns a {label: hex_color} dict for every label in `labels`.
    user_colors (dict or None) overrides auto-assigned colours per label.
    """
    color_map = {}
    palette_idx = 0
    for label in labels:
        if user_colors and label in user_colors:
            color_map[label] = user_colors[label]
        else:
            color_map[label] = _AUTO_PALETTE[palette_idx % len(_AUTO_PALETTE)]
            palette_idx += 1
    return color_map


def _is_default_labels(labels):
    return list(labels) == list(DEFAULT_LABELS)


class CategoryErrorAnalysis:
    _ALIASES = {
        'md': 'missed detection',
        'fa': 'false alarm',
    }
    _VALID_ERROR_TYPES = {'confusion', 'missed detection', 'false alarm', 'correct'}

    def __init__(self, reference, hypothesis, gender, labels=None, colors=None):
        """
        reference  : dict {uri str -> Annotation} — ground-truth diarisation
        hypothesis : dict {uri str -> Annotation} — system output
        gender     : dict {uri str -> Annotation} — category classifier output
        labels     : list of str or None
                     The complete set of category labels expected in `gender`.
                     Defaults to ["MAL", "FEM", "OCH", "KCHI"] when None.
        colors     : dict {label str -> hex str} or None
                     Per-label colour overrides.  Any label not present here
                     gets an auto-assigned colour from the internal palette.
        """
        self.reference  = reference
        self.hypothesis = hypothesis
        self.gender     = gender

        # Resolve label set (same dedup logic as CategoryStats)
        if labels is None:
            self.labels = list(DEFAULT_LABELS)
        else:
            seen = set()
            self.labels = [l for l in labels if not (l in seen or seen.add(l))]

        self.category_stats = CategoryStats(reference, gender, labels=self.labels)

        # Colour map — built once, used by all plot methods
        self._color_map = _build_color_map(
            self._plot_labels(),   # uses merged labels when default set
            user_colors=colors,
        )

        # The set of labels used for duration accumulation
        self._category_labels = set(self.labels)

        self.normalized  = True
        self.verbose     = False
        self.file_key    = None

        self.all_category_error_distribution     = {}
        self.overlap_category_error_distribution = {}
        self.all_confusion_durations             = {}
        self.overlap_confusion_durations         = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_default(self):
        return _is_default_labels(self.labels)

    def _plot_labels(self):
        """
        Returns the label list used for plotting/normalisation.
        For the default gender set OCH+KCHI are merged into CHI.
        For custom sets every label is kept as-is.
        """
        if self._is_default():
            return ["MAL", "FEM", "CHI"]
        return list(self.labels)

    def _merge(self, d):
        """
        Merges OCH+KCHI→CHI for the default label set.
        For custom label sets returns d unchanged (no merging).
        """
        if not self._is_default():
            return dict(d)
        merged = {}
        for k, v in d.items():
            if k in ("OCH", "KCHI"):
                merged["CHI"] = merged.get("CHI", 0) + v
            else:
                merged[k] = v
        return merged

    def _empty_label_dict(self):
        """Returns {label: 0.0} for every label in self.labels."""
        return dict.fromkeys(self.labels, 0.)

    # ------------------------------------------------------------------
    # Cached getters
    # ------------------------------------------------------------------

    def get_all_category_error_distribution(self, normalized=True, verbose=False, file_key=None):
        if not self.all_category_error_distribution \
                or normalized != self.normalized \
                or verbose   != self.verbose \
                or file_key  != self.file_key:
            self.all_category_error_distribution = self.category_error_distribution(
                for_overlap=False, normalized=normalized, verbose=verbose, file_key=file_key)
        return self.all_category_error_distribution

    def get_overlap_category_error_distribution(self, normalized=True, verbose=False, file_key=None):
        if not self.overlap_category_error_distribution \
                or normalized != self.normalized \
                or verbose   != self.verbose \
                or file_key  != self.file_key:
            self.overlap_category_error_distribution = self.category_error_distribution(
                for_overlap=True, normalized=normalized, verbose=verbose, file_key=file_key)
        return self.overlap_category_error_distribution

    def get_all_confusion_durations(self, file_key=None):
        if not self.all_confusion_durations or file_key != self.file_key:
            self.all_confusion_durations = self.confusion_durations(
                for_overlap=False, file_key=file_key)
        return self.all_confusion_durations

    def get_overlap_confusion_durations(self, file_key=None):
        if not self.overlap_confusion_durations or file_key != self.file_key:
            self.overlap_confusion_durations = self.confusion_durations(
                for_overlap=True, file_key=file_key)
        return self.overlap_confusion_durations

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_hypothesis(self, hypothesis):
        self.hypothesis = hypothesis

    def set_gender(self, gender):
        self.gender = gender

    def set_reference(self, reference):
        self.reference = reference

    # ------------------------------------------------------------------
    # Core analysis methods
    # ------------------------------------------------------------------

    def subset_by_error_type(self, error_annotation, error_type):
        """
        Filters an error annotation to a single error type.

        error_annotation : pyannote Annotation
        error_type       : 'confusion' | 'missed detection' (or 'md') |
                           'false alarm' (or 'fa') | 'correct'
        """
        if error_annotation is None:
            raise ValueError("error_annotation cannot be None.")

        error_type = error_type.lower()
        error_type = self._ALIASES.get(error_type, error_type)

        if error_type not in self._VALID_ERROR_TYPES:
            raise ValueError(
                f"Error type '{error_type}' must be one of {sorted(self._VALID_ERROR_TYPES)}."
            )

        filtered = error_annotation.empty()
        for segment, track, label in error_annotation.itertracks(yield_label=True):
            label_type = label[0] if isinstance(label, tuple) else label
            if label_type.lower() == error_type:
                filtered[segment, track] = label
        return filtered

    def error_category_durations(self, error_annotation, category_mapping, error_type, verbose=True):
        """
        Finds the category distribution of speech within an error annotation.

        For 'missed detection' / 'correct' : returns {label: duration} keyed by reference speaker
        For 'false alarm'                  : returns {label: duration} keyed by hypothesis speaker
        For 'confusion'                    : returns {'ref': {...}, 'pred': {...}}

        error_annotation  : pyannote Annotation
        category_mapping  : dict {speaker label -> category label} for one file
        error_type        : see subset_by_error_type
        verbose           : if True, warns about speakers with no category mapping
        """
        if error_annotation is None:
            raise ValueError("error_annotation cannot be None.")

        error_type = error_type.lower()
        error_type = self._ALIASES.get(error_type, error_type)

        if error_type not in self._VALID_ERROR_TYPES:
            raise ValueError(
                f"Error type '{error_type}' must be one of {sorted(self._VALID_ERROR_TYPES)}."
            )

        ref_durations  = self._empty_label_dict()
        pred_durations = self._empty_label_dict()
        unknown_ref    = set()
        unknown_pred   = set()

        for seg, _, label in error_annotation.itertracks(yield_label=True):
            if not isinstance(label, tuple) or len(label) < 3:
                raise ValueError(
                    f"Expected label format ('error type', 'ref speaker', 'hyp speaker'), "
                    f"got: {label!r}"
                )
            _, ref_speaker, pred_speaker = label

            if error_type in ('missed detection', 'correct'):
                ref_cat = category_mapping.get(ref_speaker)
                if ref_cat in ref_durations:
                    ref_durations[ref_cat] += seg.duration
                elif ref_speaker is not None:
                    unknown_ref.add(ref_speaker)

            elif error_type == 'false alarm':
                pred_cat = category_mapping.get(pred_speaker)
                if pred_cat in pred_durations:
                    pred_durations[pred_cat] += seg.duration
                elif pred_speaker is not None:
                    unknown_pred.add(pred_speaker)

            elif error_type == 'confusion':
                ref_cat  = category_mapping.get(ref_speaker)
                pred_cat = category_mapping.get(pred_speaker)
                if ref_cat in ref_durations:
                    ref_durations[ref_cat] += seg.duration
                elif ref_speaker is not None:
                    unknown_ref.add(ref_speaker)
                if pred_cat in pred_durations:
                    pred_durations[pred_cat] += seg.duration
                elif pred_speaker is not None:
                    unknown_pred.add(pred_speaker)

        if verbose and (unknown_ref or unknown_pred):
            parts = []
            if unknown_ref:
                parts.append(f"reference speakers with no category mapping: {sorted(unknown_ref)}")
            if unknown_pred:
                parts.append(f"hypothesis speakers with no category mapping: {sorted(unknown_pred)}")
            warnings.warn(
                f"[error_category_durations] Some speakers were skipped during "
                f"'{error_type}' duration counting.\n" + "\n".join(parts),
                UserWarning,
                stacklevel=2,
            )

        if error_type in ('missed detection', 'correct'):
            return ref_durations
        elif error_type == 'false alarm':
            return pred_durations
        else:
            return {'ref': ref_durations, 'pred': pred_durations}

    def get_overlap_fa(self, hyp, ref):
        """
        For each predicted overlap region where hyp has more speakers than ref,
        identifies false alarm speakers and sums the durations of their errors.

        hyp : pyannote Annotation for one file
        ref : pyannote Annotation for one file

        Returns: dict {predicted speaker label: total FA duration (seconds)}
        """
        mapper = HungarianMapper()
        speaker_mapping = mapper(hyp, ref)
        pred_timeline   = hyp.get_overlap()
        fa_durations    = defaultdict(float)

        for segment in pred_timeline:
            pred_speakers = set(
                spk for seg, _, spk in hyp.itertracks(yield_label=True)
                if seg & segment
            )
            ref_speakers = set(
                spk for seg, _, spk in ref.itertracks(yield_label=True)
                if seg & segment
            )

            if len(pred_speakers) <= len(ref_speakers):
                continue

            mapped_pred_speakers = {
                p_spk for p_spk, r_spk in speaker_mapping.items()
                if r_spk in ref_speakers and p_spk in pred_speakers
            }

            fa_speakers = pred_speakers - mapped_pred_speakers
            for spk in fa_speakers:
                for seg, _, s in hyp.itertracks(yield_label=True):
                    if s == spk:
                        intersection = seg & segment
                        if intersection:
                            fa_durations[spk] += intersection.duration

        return dict(fa_durations)

    def category_error_distribution(self, for_overlap=False, file_key=None,
                                    normalized=True, verbose=True):
        """
        Computes the category distribution of each error type across the dataset.

        Returns:
            {
              'md':        {label: float, ...},
              'fa':        {label: float, ...},
              'confusion': {'ref': {label: float, ...}, 'pred': {label: float, ...}},
              'correct':   {label: float, ...},
            }

        for_overlap : if True, error analysis runs only on speech-overlap regions
        file_key    : list of uri strings, or None for the full dataset
        normalized  : True  → divide by each label's total reference duration
                      False → divide by the total error duration for that error type
        verbose     : if True, prints warnings about unmapped speakers
        """
        totals = {
            'md':        {'ref':  self._empty_label_dict()},
            'fa':        {'pred': self._empty_label_dict()},
            'confusion': {'ref':  self._empty_label_dict(),
                          'pred': self._empty_label_dict()},
            'correct':   {'ref':  self._empty_label_dict()},
        }

        keys    = file_key if file_key is not None else self.reference.keys()
        mapping = self.category_stats.get_speaker_gender_map(file_key)

        for key in keys:
            analyzer = IdentificationErrorAnalysis()

            if for_overlap:
                overlap_timeline = self.reference[key].get_overlap()
                ref_overlap      = self.reference[key].crop(overlap_timeline, mode="intersection")
                pred_overlap     = self.hypothesis[key].crop(overlap_timeline, mode="intersection")
                errors           = analyzer.difference(ref_overlap, pred_overlap)
            else:
                errors = analyzer.difference(self.reference[key], self.hypothesis[key])

            error_subsets = {
                'md':        self.subset_by_error_type(errors, "missed detection"),
                'confusion': self.subset_by_error_type(errors, "confusion"),
                'correct':   self.subset_by_error_type(errors, "correct"),
            }

            category_mapping = mapping[key]

            if for_overlap:
                fa_speaker_durations = self.get_overlap_fa(
                    self.hypothesis[key], self.reference[key])
                for spk, duration in fa_speaker_durations.items():
                    cat = category_mapping.get(spk)
                    if cat in self._category_labels:
                        totals['fa']['pred'][cat] += duration
            else:
                error_subsets['fa'] = self.subset_by_error_type(errors, "false alarm")

            for error_type, subset in error_subsets.items():
                durations = self.error_category_durations(
                    subset, category_mapping, error_type, verbose)

                if error_type == 'confusion':
                    for cat in self.labels:
                        totals['confusion']['ref'][cat]  += durations['ref'][cat]
                        totals['confusion']['pred'][cat] += durations['pred'][cat]
                elif error_type == 'fa' and not for_overlap:
                    for cat in self.labels:
                        totals['fa']['pred'][cat] += durations[cat]
                else:
                    for cat in self.labels:
                        totals[error_type]['ref'][cat] += durations[cat]

        def normalise(d, cs, file_key, normalized):
            merged = self._merge(d)
            if normalized:
                merged_durations = self._merge(
                    cs.all_durations(file_key=file_key, verbose=verbose))
                return {
                    k: v / merged_durations[k]
                    for k, v in merged.items()
                    if k != "TOTAL" and merged_durations.get(k, 0) > 0
                }
            else:
                total = sum(merged.values())
                if total == 0:
                    return {k: 0. for k in merged}
                return {k: v / total for k, v in merged.items()}

        return {
            'md':        normalise(totals['md']['ref'],        self.category_stats, file_key, normalized),
            'fa':        normalise(totals['fa']['pred'],       self.category_stats, file_key, normalized),
            'confusion': {
                'ref':  normalise(totals['confusion']['ref'],  self.category_stats, file_key, normalized),
                'pred': normalise(totals['confusion']['pred'], self.category_stats, file_key, normalized),
            },
            'correct':   normalise(totals['correct']['ref'],   self.category_stats, file_key, normalized),
        }

    @staticmethod
    def diff_error_rates(a, b, round_to=None):
        def diff(x, y):
            d = x - y
            return round(d, round_to) if round_to is not None else d

        def fmt(v):
            pct  = v * 100
            sign = "+" if pct > 0 else ""
            return f"{sign}{pct:.1f}%"

        def fmt_block(d):
            return "\n".join(f"  * {label}: {fmt(v)}" for label, v in d.items())

        simple_keys = ['md', 'fa', 'correct']
        label_names = {'md': 'Missed Detections', 'fa': 'False Alarms', 'correct': 'Correct'}
        result = {}
        for k in simple_keys:
            result[k] = {label: diff(a[k][label], b[k][label]) for label in a[k]}
        result['confusion'] = {
            'ref':  {label: diff(a['confusion']['ref'][label],  b['confusion']['ref'][label])
                     for label in a['confusion']['ref']},
            'pred': {label: diff(a['confusion']['pred'][label], b['confusion']['pred'][label])
                     for label in a['confusion']['pred']},
        }

        lines = [
            "Improvements from b to a",
            "Change in proportion of each category's speech affected by each error type:\n",
        ]
        for k in simple_keys:
            lines.append(label_names[k])
            lines.append(fmt_block(result[k]))
        lines.append("Confusion (Reference speakers)")
        lines.append(fmt_block(result['confusion']['ref']))
        lines.append("Confusion (Predicted speakers)")
        lines.append(fmt_block(result['confusion']['pred']))

        print("\n".join(lines))
        return result

    def confusion_durations(self, for_overlap=False, file_key=None, ignore_unknown=False):
        """
        Finds the normalised duration for every (reference label, predicted label) confusion pair.

        For the default label set (MAL/FEM/OCH/KCHI), OCH and KCHI are merged into CHI and
        shorthand single-character keys are used (M, F, C) — identical to original behaviour.

        For custom label sets, full label strings are used as keys:
            e.g. {"LABEL_A+LABEL_A": 0.12, "LABEL_A+LABEL_B": 0.05, ...}
        The separator "+" is used to avoid ambiguity with multi-character label names.

        for_overlap    : if True, analysis is restricted to overlap regions
        file_key       : list of uri strings, or None for the full dataset
        ignore_unknown : if True, pairs where either speaker has no mapping are dropped silently
        """
        confusion_durations = defaultdict(float)
        keys            = file_key if file_key is not None else self.reference.keys()
        cs              = self.category_stats
        all_cat_maps    = cs.get_speaker_gender_map(file_key)

        using_default = self._is_default()

        # ── Normalise a raw label to its confusion-matrix key ──────────────────

        def normalise_label(label):
            """
            Default set : MAL→M, FEM→F, OCH/KCHI→C, unknown→None or "U"
            Custom set  : full label string, unknown→None or "UNKNOWN"
            """
            if using_default:
                if label in ("OCH", "KCHI"):
                    return "C"
                if label == "MAL":
                    return "M"
                if label == "FEM":
                    return "F"
                return None if ignore_unknown else "U"
            else:
                if label is None or label not in self._category_labels:
                    return None if ignore_unknown else "UNKNOWN"
                return label

        # ── Separator between ref and pred key ────────────────────────────────

        _SEP = "" if using_default else "+"

        # ── Normalise raw confusion durations by reference label totals ────────

        def normalise_confusion_durations(combination_durations, cs, file_key=None):
            durations = cs.all_durations(file_key=file_key)

            if using_default:
                chi_duration = durations.get("OCH", 0) + durations.get("KCHI", 0)
                ref_durations = {
                    "M": durations.get("MAL", 0),
                    "F": durations.get("FEM", 0),
                    "C": chi_duration,
                }
                # Default keys are always 2 chars (e.g. "MM", "MF"), ref label is first char
                return {
                    pair: c_dur / ref_durations[pair[0]]
                    for pair, c_dur in combination_durations.items()
                    if ref_durations.get(pair[0], 0) > 0
                }
            else:
                ref_durations = {k: v for k, v in durations.items() if k != "TOTAL"}
                # Custom keys use "+" separator, ref label is everything before the first "+"
                return {
                    pair: c_dur / ref_durations[pair.split("+")[0]]
                    for pair, c_dur in combination_durations.items()
                    if ref_durations.get(pair.split("+")[0], 0) > 0
                }

        # ── Accumulate confusion durations ─────────────────────────────────────

        for key in keys:
            analyzer = IdentificationErrorAnalysis()

            if for_overlap:
                overlap_timeline = self.reference[key].get_overlap()
                ref_overlap      = self.reference[key].crop(overlap_timeline, mode="intersection")
                pred_overlap     = self.hypothesis[key].crop(overlap_timeline, mode="intersection")
                errors           = analyzer.difference(ref_overlap, pred_overlap)
            else:
                errors = analyzer.difference(self.reference[key], self.hypothesis[key])

            confusion_timeline = self.subset_by_error_type(errors, "confusion")
            cat_map            = all_cat_maps[key]

            for seg, _, speaker in confusion_timeline.itertracks(yield_label=True):
                ref_label  = normalise_label(cat_map.get(speaker[1]))
                pred_label = normalise_label(cat_map.get(speaker[2]))
                if ref_label and pred_label:
                    confusion_durations[ref_label + _SEP + pred_label] += seg.duration

        return normalise_confusion_durations(confusion_durations, cs)

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot_confusion(self, for_overlap=False, ax=None):
        """
        Heatmap of normalised confusion durations.

        For the default label set the axes are [Male, Female, Child] as before.
        For custom label sets the full label strings are used.
        """
        durations = (self.get_overlap_confusion_durations()
                     if for_overlap else self.get_all_confusion_durations())

        if self._is_default():
            short_labels = ["M", "F", "C"]
            display_labels = ["Male", "Female", "Child"]
            sep = ""
        else:
            short_labels   = list(self.labels)
            display_labels = list(self.labels)
            sep = "+"

        matrix = np.array([
            [durations.get(r + sep + p, 0) for p in short_labels]
            for r in short_labels
        ])

        fmt = ".2f"
        if ax is None:
            fig, ax = plt.subplots()
        sns.heatmap(matrix, annot=True, fmt=fmt,
                    xticklabels=display_labels, yticklabels=display_labels,
                    cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference")
        if ax is None:
            plt.tight_layout()
            plt.show()

    def plot_analysis(self, verbose=False, normalized=True, file_key=None, ax=None,
                      title=None, show_legend=True, show_overlap=True,
                      show_confusion_split=True, show_unknown=False):
        """
        Bar chart of the proportion of each category's speech affected by each error type.

        verbose              : if True, prints warnings about unmapped speakers
        normalized           : if True, normalises by each label's total reference duration
        file_key             : list of uri strings, or None for the full dataset
        ax                   : external Axes object for side-by-side plots
        title                : custom plot title string
        show_legend          : if False, suppresses the legend (useful for multi-panel plots)
        show_overlap         : if True, hatches the portion occurring in overlap regions
        show_confusion_split : if True, splits confusion bars into same- vs cross-category
        show_unknown         : if True, shows a grey segment for speakers with no mapping
        """
        # ── Data ──────────────────────────────────────────────────────────────
        error_dist          = self.get_all_category_error_distribution(
            normalized=normalized, verbose=verbose, file_key=file_key)
        confusion_dist      = self.get_all_confusion_durations(file_key=file_key)
        overlap_dist        = self.get_overlap_category_error_distribution(
            normalized=normalized, verbose=verbose, file_key=file_key)
        overlap_conf_dist   = self.get_overlap_confusion_durations(file_key=file_key)

        # ── Label set for plotting ─────────────────────────────────────────────
        # For the default gender set OCH+KCHI are already merged into CHI by
        # category_error_distribution; _plot_labels() returns ["MAL","FEM","CHI"].
        # plot_labels = self._plot_labels()
        plot_labels = [
            l for l in self._plot_labels()
            if any(error_dist[err].get(l, 0) > 0 for err in ('md', 'fa'))
            or error_dist['confusion']['ref'].get(l, 0) > 0
        ]

        # ── Parameters ────────────────────────────────────────────────────────
        SHOW_CONFUSION_SPLIT = show_confusion_split
        SHOW_OVERLAP_HATCH   = show_overlap
        SHOW_UNKNOWN         = show_unknown
        SHOW_LEGEND          = show_legend

        ERR_TYPES = ['md', 'fa', 'confusion']

        # Separator used in confusion_durations keys
        _SEP = "" if self._is_default() else "+"

        BASE_COLORS   = self._color_map
        UNKNOWN_COLOR = '#999999'
        LIGHT_FACTOR  = 0.35
        DARK_FACTOR   = 0.35
        HATCH_SAME    = '//'

        # ── Colour helpers ─────────────────────────────────────────────────────

        def lighten(hex_color, factor=LIGHT_FACTOR):
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            return '#{:02x}{:02x}{:02x}'.format(
                int(r + (255 - r) * factor),
                int(g + (255 - g) * factor),
                int(b + (255 - b) * factor),
            )

        def darken(hex_color, factor=DARK_FACTOR):
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            return '#{:02x}{:02x}{:02x}'.format(
                int(r * (1 - factor)),
                int(g * (1 - factor)),
                int(b * (1 - factor)),
            )

        # ── Confusion split helpers ────────────────────────────────────────────

        # After the plot_labels definition, add a key-label lookup
        if self._is_default():
            _confusion_key = {"MAL": "M", "FEM": "F", "CHI": "C"}
        else:
            _confusion_key = {l: l for l in plot_labels}

        def same_frac(label, dist):
            k = _confusion_key[label]
            return dist.get(k + _SEP + k, 0.0)

        def cross_frac(label, dist):
            k = _confusion_key[label]
            prefix = k + _SEP
            unknown_suffix = _SEP + ("UNKNOWN" if _SEP else "U")
            return sum(
                v for key, v in dist.items()
                if key.startswith(prefix)
                and not key.endswith(unknown_suffix)
                and key != k + _SEP + k
            )

        def unknown_frac(label, dist):
            k = _confusion_key[label]
            unknown_key = k + _SEP + ("UNKNOWN" if _SEP else "U")
            return dist.get(unknown_key, 0.0)

        # ── Label helpers ──────────────────────────────────────────────────────

        def sub_label(ax, x, y_bottom, height, pct, bar_w, side=None, color='#333333'):
            if height < 0.003:
                return
            ax.text(x, y_bottom + height + 0.001, f'{pct * 100:.1f}%',
                    ha='center', va='bottom', fontsize=7, color='white', zorder=6)

        def side_sub_label(ax, x, y_bottom, height, pct, bar_w, dx=0.015, color='#333333'):
            if height < 1e-4:
                return
            ax.text(x + bar_w / 2 + dx, y_bottom + height / 2,
                    f'{pct * 100:.1f}%',
                    ha='left', va='center', fontsize=8.5, color=color, zorder=6)

        # ── Layout ────────────────────────────────────────────────────────────

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))

        n_groups      = len(ERR_TYPES)
        n_labels      = len(plot_labels)
        group_w       = 1.2
        intra_gap     = 0.5
        bar_w         = group_w / (n_labels + (n_labels - 1) * intra_gap)
        group_spacing = 1.8
        group_centers = np.arange(n_groups) * group_spacing

        # ── Draw bars ─────────────────────────────────────────────────────────

        for li, label in enumerate(plot_labels):
            color     = BASE_COLORS.get(label, _AUTO_PALETTE[li % len(_AUTO_PALETTE)])
            light_col = lighten(color)
            dark_col  = darken(color)
            step      = bar_w * (1 + intra_gap)
            offset    = (li - (n_labels - 1) / 2) * step
            xs        = group_centers + offset

            for ei, err in enumerate(ERR_TYPES):
                x = xs[ei]

                # ── confusion ─────────────────────────────────────────────────
                if err == 'confusion':
                    total_conf_raw = error_dist['confusion']['ref'].get(label, 0)

                    if SHOW_CONFUSION_SPLIT:
                        same_raw    = same_frac(label, confusion_dist)
                        cross_raw   = cross_frac(label, confusion_dist)
                        unknown_raw = unknown_frac(label, confusion_dist)
                        denom_raw   = same_raw + cross_raw + unknown_raw

                        scale     = total_conf_raw / denom_raw if denom_raw > 0 else 0
                        same_h    = same_raw    * scale
                        cross_h   = cross_raw   * scale
                        unknown_h = unknown_raw * scale

                        if not SHOW_UNKNOWN:
                            same_h    = same_raw  * scale
                            cross_h   = cross_raw * scale
                            unknown_h = 0.0

                        ax.bar(x, same_h, width=bar_w * 0.9,
                               color=light_col, edgecolor='white', linewidth=0.4, zorder=3)
                        side_sub_label(ax, x, 0, same_h, same_h, bar_w, color=light_col)

                        ax.bar(x, cross_h, width=bar_w * 0.9, bottom=same_h,
                               color=dark_col, edgecolor='white', linewidth=0.4, zorder=3)
                        side_sub_label(ax, x, same_h, cross_h, cross_h, bar_w, color=dark_col)

                        if SHOW_UNKNOWN and unknown_h > 0:
                            ax.bar(x, unknown_h, width=bar_w * 0.9,
                                   bottom=same_h + cross_h,
                                   color=UNKNOWN_COLOR, edgecolor='white', linewidth=0.4, zorder=3)
                            side_sub_label(ax, x, same_h + cross_h, unknown_h, unknown_h,
                                           bar_w, color=UNKNOWN_COLOR)

                        if SHOW_OVERLAP_HATCH:
                            ol_total       = overlap_dist['confusion']['ref'].get(label, 0)
                            ol_same_raw    = same_frac(label, overlap_conf_dist)
                            ol_cross_raw   = cross_frac(label, overlap_conf_dist)
                            ol_unknown_raw = unknown_frac(label, overlap_conf_dist)
                            ol_denom_raw   = ol_same_raw + ol_cross_raw + ol_unknown_raw

                            ol_scale   = ol_total / ol_denom_raw if ol_denom_raw > 0 else 0
                            ol_same_h  = ol_same_raw  * ol_scale
                            ol_cross_h = ol_cross_raw * ol_scale
                            ol_unknown_h = ol_unknown_raw * ol_scale

                            if not SHOW_UNKNOWN:
                                ol_known_raw = ol_same_raw + ol_cross_raw
                                ol_scale_k   = ol_total / ol_known_raw if ol_known_raw > 0 else 0
                                ol_same_h    = ol_same_raw  * ol_scale_k
                                ol_cross_h   = ol_cross_raw * ol_scale_k
                                ol_unknown_h = 0.0

                            if ol_same_h > 0:
                                ax.bar(x, ol_same_h, width=bar_w * 0.9, bottom=0,
                                       color=light_col, edgecolor='white',
                                       hatch=HATCH_SAME, linewidth=0.5, zorder=4)
                                sub_label(ax, x, 0, ol_same_h, ol_same_h, bar_w, color=color)

                            if ol_cross_h > 0:
                                ax.bar(x, ol_cross_h, width=bar_w * 0.9, bottom=same_h,
                                       color=dark_col, edgecolor='white',
                                       hatch=HATCH_SAME, linewidth=0.5, zorder=4)
                                sub_label(ax, x, same_h, ol_cross_h, ol_cross_h, bar_w, color=color)

                            if SHOW_UNKNOWN and ol_unknown_h > 0:
                                ax.bar(x, ol_unknown_h, width=bar_w * 0.9,
                                       bottom=same_h + cross_h,
                                       color=UNKNOWN_COLOR, edgecolor='white',
                                       hatch=HATCH_SAME, linewidth=0.5, zorder=4)
                                sub_label(ax, x, same_h + cross_h, ol_unknown_h, ol_unknown_h,
                                          bar_w, color=UNKNOWN_COLOR)

                    else:
                        if SHOW_UNKNOWN:
                            bar_val = total_conf_raw
                        else:
                            unknown_raw = unknown_frac(label, confusion_dist)
                            denom_raw   = (same_frac(label, confusion_dist)
                                           + cross_frac(label, confusion_dist)
                                           + unknown_raw)
                            unknown_h   = (unknown_raw / denom_raw * total_conf_raw
                                           if denom_raw > 0 else 0)
                            bar_val     = total_conf_raw - unknown_h

                        ax.bar(x, bar_val, width=bar_w * 0.9,
                               color=color, edgecolor='white', linewidth=0.4, zorder=3)

                        if SHOW_OVERLAP_HATCH:
                            ol_val = overlap_dist['confusion']['ref'].get(label, 0)
                            if ol_val > 0:
                                ax.bar(x, ol_val, width=bar_w * 0.9, bottom=0,
                                       color=color, edgecolor='white',
                                       hatch=HATCH_SAME, linewidth=0.5, zorder=4)

                    bar_top = (same_h + cross_h + (unknown_h if SHOW_UNKNOWN else 0)
                               if SHOW_CONFUSION_SPLIT else bar_val)

                # ── md / fa ───────────────────────────────────────────────────
                else:
                    val = error_dist[err].get(label, 0)
                    ax.bar(x, val, width=bar_w * 0.9,
                           color=color, edgecolor='white', linewidth=0.4, zorder=3)

                    if SHOW_OVERLAP_HATCH and err in ('md', 'fa'):
                        ol_val = overlap_dist[err].get(label, 0)
                        if ol_val > 0:
                            ax.bar(x, ol_val, width=bar_w * 0.9, bottom=0,
                                   color=color, edgecolor='white',
                                   hatch=HATCH_SAME, linewidth=0.5, zorder=4)
                            sub_label(ax, x, 0, ol_val, ol_val, bar_w, color=color)

                    bar_top = val

                ax.text(x, bar_top + 0.001, f'{bar_top * 100:.1f}%',
                        ha='center', va='bottom', fontsize=9,
                        color='#333333', fontweight='bold', zorder=5)

        # ── Legend ────────────────────────────────────────────────────────────

        legend_handles = []

        # One colour patch per label — use display name from plot_labels
        display_names = (
            {'MAL': 'Male', 'FEM': 'Female', 'CHI': 'Child'}
            if self._is_default()
            else {l: l for l in plot_labels}
        )
        for label in plot_labels:
            legend_handles.append(
                mpatches.Patch(color=BASE_COLORS.get(label, '#888888'),
                               label=display_names.get(label, label)))

        if SHOW_CONFUSION_SPLIT:
            legend_handles += [
                mpatches.Patch(color='#cccccc', label='Same-category confusion'),
                mpatches.Patch(color='#333333', label='Cross-category confusion'),
            ]
            if SHOW_UNKNOWN:
                legend_handles.append(
                    mpatches.Patch(color=UNKNOWN_COLOR, label='Unknown speaker confusion'))

        if SHOW_OVERLAP_HATCH:
            legend_handles.append(
                mpatches.Patch(facecolor='none', edgecolor='#444444',
                               hatch=HATCH_SAME, label='Occurring in overlap regions'))

        if SHOW_LEGEND:
            ax.legend(handles=legend_handles, loc='upper left',
                      bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                      framealpha=0.9, fontsize=9,
                      title='Legend', title_fontsize=9)

        # ── Axes cosmetics ─────────────────────────────────────────────────────

        ax.set_xticks(group_centers)
        ax.set_xticklabels(['Missed Detection', 'False Alarm', 'Confusion'], fontsize=11)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(axis='y', which='major', linestyle='-',  linewidth=0.5, alpha=0.7, zorder=0)
        ax.grid(axis='y', which='minor', linestyle='--', linewidth=0.3, alpha=0.7, zorder=0)
        ax.set_ylabel("% of category's total speech duration", fontsize=10)
        title = title if title else 'Proportion of Category Speech Affected by Error Type'
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        padding = 0.6
        ax.set_xlim(group_centers[0] - padding, group_centers[-1] + padding)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ax is None:
            plt.tight_layout()
            plt.show()