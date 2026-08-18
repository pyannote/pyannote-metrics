"""
Metrics computation for category-based error analysis.

Two classes:
  - CategoryStats: reference-only stats (speech duration per category label).
  - CategoryErrorAnalysis: DER breakdown per category label.

This module has no plotting dependency. See category_plots.py for visualizing
the resulting durations / error-distribution attributes.

---------------------------------------------------------------------------
Example usage
---------------------------------------------------------------------------

files = [
    {
        "uri": "DH_EVAL_0001",
        "reference": <ground-truth speaker Annotation>,
        "hypothesis": <hypothesis speaker Annotation>,
        "gender": <Annotation with 'male'/'female' segments>,
    },
    ...
]

--- CategoryStats: reference-only stats, independent of any hypothesis ---

stats = CategoryStats(files=files, speaker_annotation="reference",
                       speaker_category_map_key="speaker_category_map",
                       category_annotation="gender")

stats.compute_durations(overlap=True)      # populates stats.durations, stats.durations_overlap
stats.durations                             # {'male': 120.3, 'female': 98.7, 'TOTAL': 219.0}
stats.label_duration("female")              # 98.7
stats.label_percentage("female")            # 0.45
stats.label_ratio("male", "female")         # 1.22
stats.print_summary()

stats.compute_durations(overlap=False, uris=["DH_EVAL_0001"])   # recompute for a subset

stats.get_missing_speaker_labels(verbose=True)   # diagnostic: speakers with no category mapping
stats.get_category_classifier_accuracy()         # prints MD/FA of the category labels themselves

--- Per-file pre-computed speaker to category mapping ---

Sometimes you already know the category of every speaker in a file (e.g.
from a diarization-independent classifier, manual labels, or metadata) and
want to skip the overlap-based 'map_speaker_to_category' step for that file
entirely. Each file dict may carry its own optional key holding that file's {speaker_label: category_label}
mapping. For example:

files = [
    {
        "uri": "DH_EVAL_0001",
        "reference": ann1,
        "speaker_category_map": {"spk1": "male", "spk2": "female"},
        # no "gender" key needed -- this file is fully covered by its own map
    },
    {
        "uri": "DH_EVAL_0002",
        "reference": ann2,
        "gender": gender_ann2,
        # no "speaker_category_map" key -- falls back to computing map from gender annotation
    },
]

stats = CategoryStats(files=files, speaker_annotation="reference",
                       category_annotation="gender", speaker_category_map_key="speaker_category_map")

If a file's dict contains a speaker category map, that mapping is
used as-is for that file. Otherwise the file requires an annotation.

Everything downstream (compute_durations, print_summary, merge_labels,
exclude_labels, and CategoryErrorAnalysis's error breakdowns) works
identically regardless of whether a given file's mapping came from overlap
or was supplied directly.

--- CategoryErrorAnalysis: hypothesis-vs-reference error breakdown per category ---

analysis = CategoryErrorAnalysis(files=files, speaker_annotation="reference",
                                  hypothesis_annotation="hypothesis",
                                  speaker_category_map_key="speaker_category_map",
                                  category_annotation="gender")

analysis.compute_distributions(normalized=True)   # populates the four error category attributes
analysis.error_distribution         # {'md': {...}, 'fa': {...}, 'confusion': {...}, 'correct': {...}}
analysis.confusion_durations        # {'male+female': 0.03, 'female+male': 0.02, ...}
analysis.overlap_error_distribution # {'md': {...}, 'fa': {...}, 'confusion': {...}, 'correct': {...}}
analysis.overlap_confusion_durations # {'male+female': 0.01, 'female+male': 0.01, ...}

# category_annotation is only skippable if EVERY file supplies its own map --
# it's still "one source or the other per file", never "neither". Here both
# files carry their own precomputed map, so no annotation key is needed at all:
files_fully_precomputed = [
    {"uri": "DH_EVAL_0001", "reference": ann1, "hypothesis": hyp1,
     "speaker_category_map": {"spk1": "male", "spk2": "female"}},
    {"uri": "DH_EVAL_0002", "reference": ann2, "hypothesis": hyp2,
     "speaker_category_map": {"spk1": "female", "spk2": "female", "spk3": "male"}},
]
analysis = CategoryErrorAnalysis(files=files_fully_precomputed, speaker_annotation="reference",
                                  hypothesis_annotation="hypothesis", speaker_category_map_key="speaker_category_map")

--- Comparing two models' distributions ---

analysis.set_hypothesis("community")
analysis.compute_distributions(normalized=True)
community_dist = analysis.error_distribution

analysis.set_hypothesis("precision")
analysis.compute_distributions(normalized=True)
precision_dist = analysis.error_distribution

CategoryErrorAnalysis.diff_error_rates(community_dist, precision_dist)   # prints + returns the deltas

--- Extracting plain dicts for anything other than plotting ---

md_subset = analysis.subset_by_error_type(analysis._get_or_compute_diff("DH_EVAL_0001"), "missed detection")
durations_by_category = analysis.error_category_durations(
    md_subset, analysis.category_stats.speaker_category_map["DH_EVAL_0001"], "missed detection")
"""

import warnings
from collections import defaultdict

from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.errors.identification import IdentificationErrorAnalysis
from pyannote.metrics.matcher import HungarianMapper


class CategoryStats:
    """
    Provides statistics about reference annotations with respect to given
    categorical attributes of the speakers (e.g. gender, age bracket, accent
    distribution).

    Each file is represented as a plain dict with at least these keys:
      - uri key        : a string identifying the audio file
      - speaker key    : a pyannote Annotation with ground-truth speaker-identity labels

    Category resolution happens per file, using one of two sources:
      - category key   : a pyannote Annotation with attribute labels (gender, age
                          bracket, …). Used via temporal overlap with the speaker
                          annotation when the file has no pre-computed map (see below).
      - map key         : an optional {speaker_label: category_label} dict. When present on a file, it is used
                          as-is for that file and no overlap computation is performed.

    Every file must be resolvable by at least one of these two sources. A file
    with neither is a construction-time error.

    Label sets are discovered automatically from the data (from category
    annotations and/or pre-computed maps, depending on what each file has).
    merge_labels and exclude_labels are applied during discovery.

    Public API surface:
      - compute_speaker_category_map(uris=None) -> populates self.speaker_category_map
        (per file: from its own pre-computed map if present, else derived from
        temporal overlap with category_annotation)
      - compute_durations(overlap=True, uris=None, verbose=True) -> populates
        self.durations ({label: seconds, …, 'TOTAL': seconds}) and self.durations_overlap
      - label_duration / all_durations / label_percentage / all_percentages /
        label_ratio / print_summary: read self.durations (call compute_durations() first)
      - get_missing_speaker_labels(uris=None, verbose=False): diagnostic, always fresh
      - get_category_classifier_accuracy(uris=None): diagnostic, prints, not cached;
        requires an actual category_annotation for every requested uri (raises for
        any uri that only has a pre-computed map)
    """

    def __init__(
        self,
        files,
        speaker_annotation,
        speaker_category_map_key,
        category_annotation=None,
        uri_key='uri',
        merge_labels=None,
        exclude_labels=None,
    ):
        """
        files                     : list of dicts, one per audio file
        speaker_annotation        : str — key in each dict for the speaker diarization Annotation
        speaker_category_map_key  : str — key under which an individual file dict may optionally
                                     carry a pre-computed {speaker_label: category_label} mapping.
                                     When a file has this key, that mapping is used as-is (after
                                     merge/exclude resolution) for that file, instead of computing
                                     overlap between speaker_annotation and category_annotation.
                                     A file without this key falls back to overlap and therefore
                                     must carry category_annotation. Required (no default) so that
                                     the key name is always explicit at the call site, since it's
                                     read from arbitrary user file dicts.
        category_annotation       : str or None — key in each dict for the attribute/category
                                     Annotation. Only required for files that don't carry their
                                     own pre-computed map.
        uri_key                   : str — key in each dict for the audio URI (default: 'uri')
        merge_labels              : dict {new_label: (old_label, …)} or None to either merge or rename labels
        exclude_labels            : list of str or None — labels to drop from analysis

        Example
        -------
        stats = CategoryStats(files=files, speaker_annotation="reference",
                               speaker_category_map_key="speaker_category_map",
                               category_annotation="gender")
        stats.compute_durations(overlap=True)
        stats.print_summary()

        # Or, with some files carrying their own pre-computed mapping:
        # files = [{"uri": "A", "reference": ann, "speaker_category_map": {"spk1": "male"}}, ...]
        stats = CategoryStats(files=files, speaker_annotation="reference",
                               speaker_category_map_key="speaker_category_map")
        """
        self.files = files
        self.speaker_annotation_key = speaker_annotation
        self.category_annotation_key = category_annotation
        self.uri_key = uri_key
        self.merge_labels = merge_labels or {}
        self.exclude_labels = set(exclude_labels or [])
        self.speaker_category_map_key = speaker_category_map_key

        # Convenience look-ups so the rest of the class can access annotations
        # by URI without iterating self.files every time.
        self._speaker = {f[uri_key]: f[speaker_annotation] for f in files}

        # Category annotation is only collected for files that actually carry
        # it -- some files may rely entirely on a pre-computed map instead.
        self._category = {
            f[uri_key]: f[category_annotation]
            for f in files
            if category_annotation is not None and category_annotation in f
        }

        # Pre-computed per-file mapping, in canonical form
        # {uri: {speaker_label: raw_category_label}}. Only present for files
        # that actually carry the key (with a non-None value).
        self._precomputed_map = {
            f[uri_key]: dict(f[speaker_category_map_key])
            for f in files
            if speaker_category_map_key in f and f[speaker_category_map_key] is not None
        }

        # Every file must be resolvable via one source or the other.
        unresolved = [
            uri for uri in self._speaker
            if uri not in self._precomputed_map and uri not in self._category
        ]
        if unresolved:
            raise ValueError(
                f"{len(unresolved)} file(s) have neither a '{speaker_category_map_key}' "
                f"entry nor a usable '{category_annotation}' category_annotation to fall "
                f"back on: {sorted(unresolved)}"
            )

        # Labels are discovered from the data; merge/exclude applied in place.
        self.labels = self._discover_labels()

        # Overlap-coverage warning only makes sense for files whose category
        # comes from an actual temporal category annotation; files with a
        # pre-computed map are skipped inside this method.
        self._warn_unknown_category_labels()

        # Populated by compute_speaker_category_map() / compute_durations().
        self.speaker_category_map = None
        self.durations = None
        self.durations_overlap = None

    # ------------------------------------------------------------------
    # Label discovery
    # ------------------------------------------------------------------

    def _discover_labels(self):
        """
        Collects every label that appears in the data, per file: from that
        file's pre-computed map values if it has one, otherwise from its
        category annotation's track labels. Applies exclusions, then renames
        according to merge_labels. Returns a deduplicated, ordered list.
        """
        seen = []
        seen_set = set()

        for uri in self._speaker:
            if uri in self._precomputed_map:
                label_source = self._precomputed_map[uri].values()
            else:
                label_source = (
                    label
                    for _, _, label in self._category[uri].itertracks(yield_label=True)
                )
            for label in label_source:
                if label not in seen_set:
                    seen.append(label)
                    seen_set.add(label)

        # Apply exclusions
        active = [l for l in seen if l not in self.exclude_labels]

        # Apply merges: replace old label names with the new consolidated name.
        # Build a reverse map: old_label -> new_label
        rename = {}
        for new_label, old_labels in self.merge_labels.items():
            for old in old_labels:
                rename[old] = new_label

        merged = []
        merged_set = set()
        for label in active:
            canonical = rename.get(label, label)
            if canonical not in merged_set:
                merged.append(canonical)
                merged_set.add(canonical)

        return merged

    def _resolve_category_label(self, raw_label):
        """Maps a raw label from the data to its canonical (post-merge) form,
        or returns None if the label is excluded."""
        if raw_label in self.exclude_labels:
            return None
        for new_label, old_labels in self.merge_labels.items():
            if raw_label in old_labels:
                return new_label
        return raw_label

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    def _warn_unknown_category_labels(self):
        """
        Warns about speakers in the speaker annotation that have no temporal
        overlap with any category segment and therefore cannot be mapped to
        any label. These speakers will be skipped during duration
        and error computations, but their duration can be found in the
        inter/intra category confusion distribution under "UNKNOWN".

        Files with a pre-computed map are skipped entirely here: overlap
        coverage is only a meaningful diagnostic for files whose category
        comes from an actual temporal category annotation.
        """
        unmapped = defaultdict(set)

        for uri in self._speaker:
            if uri in self._precomputed_map:
                continue

            speaker_ann  = self._speaker[uri]
            category_ann = self._category[uri]

            for seg, _, speaker in speaker_ann.itertracks(yield_label=True):
                has_overlap = any(
                    (seg & cat_seg).duration > 0
                    for cat_seg, _, _ in category_ann.itertracks(yield_label=True)
                )
                if not has_overlap:
                    unmapped[uri].add(speaker)

        if unmapped:
            total = sum(len(v) for v in unmapped.values())
            lines = [f"  {uri}: {sorted(v)}" for uri, v in sorted(unmapped.items())]
            warnings.warn(
                f"{total} speaker(s) across {len(unmapped)} file(s) have no overlap "
                f"with any category segment and will be excluded from all computations:\n"
                + "\n".join(lines),
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def map_speaker_to_category(speaker_segments, category_segments):
        """
        Maps each speaker in a single file to the category label with the
        greatest temporal overlap.

        speaker_segments  : Annotation — speaker-labelled annotation for one file
        category_segments : Annotation — attribute-labelled annotation for one file

        Returns {speaker_label: category_label}
        """
        votes = defaultdict(list)

        for seg, _, speaker in speaker_segments.itertracks(yield_label=True):
            for cat_seg, _, category in category_segments.itertracks(yield_label=True):
                overlap = seg & cat_seg
                if overlap.duration > 0:
                    votes[speaker].append((category, overlap.duration))

        mapping = {}
        for speaker, speaker_votes in votes.items():
            weights = defaultdict(float)
            for category, duration in speaker_votes:
                weights[category] += duration
            mapping[speaker] = max(weights, key=weights.get)

        return mapping

    # ------------------------------------------------------------------
    # Accuracy / error metrics
    # ------------------------------------------------------------------

    def get_category_classifier_accuracy(self, uris=None):
        """
        Prints missed detections and false alarms of the category classifier
        against the speaker reference.

        Requires an actual category_annotation for every requested uri (this
        compares the reference speaker timeline against the category timeline
        itself, which has no meaning for a file that only has a pre-computed
        speaker_category_map).

        uris : list of URI strings to restrict analysis, or None for all
        """
        keys = uris if uris is not None else list(self._speaker.keys())
        missing = [k for k in keys if k not in self._category]
        if missing:
            raise RuntimeError(
                "get_category_classifier_accuracy() requires an actual category "
                f"annotation; these file(s) only have a pre-computed map (or "
                f"neither): {sorted(missing)}"
            )
        metric = DiarizationErrorRate()
        for key in keys:
            metric(self._speaker[key], self._category[key], detailed=True)
        print(f"Missed Detections: {metric['missed detection'] / metric['total']:.4f}")
        print(f"False Alarms:      {metric['false alarm'] / metric['total']:.4f}")

    # ------------------------------------------------------------------
    # Speaker → category mapping
    # ------------------------------------------------------------------

    def compute_speaker_category_map(self, uris=None):
        """
        Maps every speaker to their most likely category label and stores the
        result on self.speaker_category_map. Always recomputes (no caching)
        -- call once per 'uris' subset you need.

        Resolution is per file: a file's own pre-computed map is used directly
        (after merge/exclude resolution) if present; otherwise the mapping is
        derived from temporal overlap with that file's category_annotation.

        Returns {uri: {speaker_label: category_label}}

        uris : list of URI strings, or None for all
        """
        keys = uris if uris is not None else self._speaker.keys()
        mapping = {}

        for key in keys:
            if key in self._precomputed_map:
                raw_map = self._precomputed_map[key]
            else:
                raw_map = self.map_speaker_to_category(
                    self._speaker[key], self._category[key]
                )

            resolved = {}
            for speaker, raw_label in raw_map.items():
                canonical = self._resolve_category_label(raw_label)
                if canonical is not None:
                    resolved[speaker] = canonical
            mapping[key] = resolved

        self.speaker_category_map = mapping
        return self.speaker_category_map

    def get_missing_speaker_labels(self, uris=None, verbose=False):
        """
        Returns speakers that could not be mapped to any category label.
        Always recomputes the mapping fresh for the requested 'uris' (this is
        a diagnostic helper, not part of the compute_durations() flow, so it
        doesn't touch self.speaker_category_map's other cached use).

        Works identically whether a given file's mapping comes from temporal
        overlap or from its own pre-computed map.

        Output: {uri: set of unmapped speaker label strings}

        uris    : list of URI strings, or None for all
        verbose : if True, prints a summary
        """
        mapping = self.compute_speaker_category_map(uris=uris)
        keys = uris if uris is not None else self._speaker.keys()

        missing = {}
        for key in keys:
            unmapped = {
                speaker
                for _, _, speaker in self._speaker[key].itertracks(yield_label=True)
                if speaker not in mapping[key]
            }
            if unmapped:
                missing[key] = unmapped

        if verbose and missing:
            total_speakers = sum(len(v) for v in missing.values())
            total_duration = sum(
                seg.duration
                for key in missing
                for seg, _, speaker in self._speaker[key].itertracks(yield_label=True)
                if speaker in missing[key]
            )
            lines = [f"  {k}: {sorted(v)}" for k, v in sorted(missing.items())]
            print(
                f"{total_speakers} speaker(s) across {len(missing)} file(s) "
                f"could not be mapped to a category, accounting for "
                f"{total_duration:.2f}s of untracked duration:\n"
                + "\n".join(lines)
            )

        return missing

    # ------------------------------------------------------------------
    # Duration computation
    # ------------------------------------------------------------------

    def compute_durations(self, overlap=True, uris=None, verbose=True):
        """
        Computes cumulative speech duration per category label and stores the
        result on self.durations, along with the overlap mode used on
        self.durations_overlap (read by category_plots.py to pick title
        wording). Always recomputes from scratch (no caching) -- call once
        per (overlap, uris) combination you need.

        Note: the speaker->category mapping used here is always computed over
        ALL files (not restricted to 'uris'), even when 'uris' restricts which
        files' durations get summed into the result.

        Returns {label: float, …, 'TOTAL': float}

        uris    : list of URI strings to restrict which files' durations get
                  summed, or None for all
        overlap : True  → TOTAL counts overlapping speech
                  False → TOTAL is the support (non-overlapping) timeline length
        verbose : if True, warns about unmapped speakers
        """
        self.compute_speaker_category_map()
        mapping = self.speaker_category_map

        keys = uris if uris is not None else self._speaker.keys()
        durations = {label: 0.0 for label in self.labels}
        unknown_speakers = defaultdict(set)

        for key in keys:
            for seg, _, speaker in self._speaker[key].itertracks(yield_label=True):
                category_label = mapping[key].get(speaker)
                if category_label is None or category_label not in durations:
                    unknown_speakers[key].add(speaker)
                else:
                    durations[category_label] += seg.duration

        if overlap:
            total = sum(durations[l] for l in self.labels)
        else:
            total = sum(
                segment.duration
                for key in keys
                for segment in self._speaker[key].get_timeline().support()
            )

        durations['TOTAL'] = total

        if unknown_speakers and verbose:
            total_unknown = sum(len(v) for v in unknown_speakers.values())
            total_unknown_duration = sum(
                seg.duration
                for key in unknown_speakers
                for seg, _, speaker in self._speaker[key].itertracks(yield_label=True)
                if speaker in unknown_speakers[key]
            )
            lines = [f"  {k}: {sorted(v)}" for k, v in sorted(unknown_speakers.items())]
            warnings.warn(
                f"{total_unknown} speaker(s) across {len(unknown_speakers)} file(s) "
                f"could not be mapped to a category, accounting for "
                f"{total_unknown_duration:.2f}s of untracked duration:\n"
                + "\n".join(lines),
                UserWarning,
                stacklevel=3,
            )

        self.durations = durations
        self.durations_overlap = overlap
        return self.durations

    # ------------------------------------------------------------------
    # Public duration / percentage API
    # ------------------------------------------------------------------

    def _require_durations(self):
        if self.durations is None:
            raise RuntimeError(
                "Call compute_durations() before reading duration/percentage results."
            )

    def label_duration(self, label):
        """Returns the total speech duration (seconds) for a single label."""
        self._require_durations()
        return self.durations[label]

    def all_durations(self):
        """Returns the full {label: duration, 'TOTAL': duration} dict."""
        self._require_durations()
        return self.durations

    def label_percentage(self, label):
        """Returns the fraction of total time attributed to 'label'."""
        self._require_durations()
        d = self.durations
        return 0.0 if d['TOTAL'] == 0 else d[label] / d['TOTAL']

    def all_percentages(self):
        """Returns {label: fraction} for every label in self.labels (TOTAL excluded)."""
        self._require_durations()
        d = self.durations
        if d['TOTAL'] == 0:
            return {label: 0.0 for label in self.labels}
        return {label: d[label] / d['TOTAL'] for label in self.labels}

    def label_ratio(self, label_a, label_b):
        """Returns the ratio of label_a duration to label_b duration."""
        self._require_durations()
        d = self.durations
        for label in (label_a, label_b):
            if label not in d:
                raise ValueError(f"Unknown label '{label}'. Valid: {list(d)}")
        if d[label_b] == 0:
            raise ValueError(f"Duration of '{label_b}' is zero; cannot compute ratio.")
        return d[label_a] / d[label_b]

    def print_summary(self):
        """Prints per-label durations and their percentage of the total."""
        self._require_durations()
        d = self.durations
        for label in self.labels:
            pct = 0.0 if d['TOTAL'] == 0 else d[label] / d['TOTAL']
            print(f"{label}: {d[label]:.2f}s ({pct:.1%} of total)")
        print(f"TOTAL: {d['TOTAL']:.2f}s")


class CategoryErrorAnalysis:
    """
    Error analysis of a speaker diarization model with respect to given
    categorical attributes of the segments or speakers (e.g. gender, age
    bracket, accent distribution).

    Each file is a plain dict containing at least:
      - uri key        : string identifying the audio file
      - reference key  : pyannote Annotation with ground-truth speaker-identity labels
      - hypothesis key : pyannote Annotation with predicted speaker labels

    Category resolution (per file) is delegated to an internal CategoryStats
    instance: a file uses its own pre-computed speaker_category_map if it has
    one, otherwise falls back to temporal overlap with category_annotation.
    See CategoryStats for full details. Label sets are discovered
    automatically from that data. merge_labels and exclude_labels are applied
    at construction time and propagated to it.

    Public API surface:
      - compute_distributions(normalized=True, verbose=True, uris=None) ->
        populates self.error_distribution, self.overlap_error_distribution,
        self.confusion_durations, self.overlap_confusion_durations
      - self.labels: label list (from internal CategoryStats)
      - subset_by_error_type / error_category_durations / get_overlap_fa /
        diff_error_rates: usable standalone for custom analysis/export
    """

    _ALIASES = {'md': 'missed detection', 'fa': 'false alarm'}
    _VALID_ERROR_TYPES = {'confusion', 'missed detection', 'false alarm', 'correct'}

    def __init__(
        self,
        files,
        speaker_annotation,
        hypothesis_annotation,
        speaker_category_map_key,
        category_annotation=None,
        uri_key='uri',
        merge_labels=None,
        exclude_labels=None,
    ):
        """
        files                      : list of dicts, one per audio file
        speaker_annotation         : str — key for the ground-truth speaker Annotation
        hypothesis_annotation      : str — key for a speaker diarization hypothesis Annotation
        speaker_category_map_key   : str — key under which an individual file dict may
                                      optionally carry a pre-computed
                                      {speaker_label: category_label} mapping. Forwarded
                                      to the internal CategoryStats; see CategoryStats for
                                      the per-file resolution rules. Required (no default)
                                      so the key name is always explicit at the call site.
        category_annotation        : str or None — key for the attribute/category Annotation.
                                      Only required for files that don't carry their own
                                      pre-computed map (see speaker_category_map_key).
        uri_key                    : str — key for the audio URI/ID (default: 'uri')
        merge_labels               : dict {new_label: (old_label, …)} or None
        exclude_labels              : list of str or None

        Example
        -------
        analysis = CategoryErrorAnalysis(files=files, speaker_annotation="reference",
                                          hypothesis_annotation="precision",
                                          speaker_category_map_key="speaker_category_map",
                                          category_annotation="gender")
        analysis.compute_distributions()
        analysis.error_distribution        # {'md': {...}, 'fa': {...}, 'confusion': {...}, 'correct': {...}}

        # Or with some files carrying their own pre-computed mapping and others
        # falling back to overlap with "gender":
        analysis = CategoryErrorAnalysis(files=files, speaker_annotation="reference",
                                          hypothesis_annotation="precision",
                                          speaker_category_map_key="speaker_category_map",
                                          category_annotation="gender")
        """
        self.files = files
        self.speaker_annotation_key    = speaker_annotation
        self.hypothesis_annotation_key = hypothesis_annotation
        self.category_annotation_key   = category_annotation
        self.uri_key         = uri_key
        self.merge_labels    = merge_labels or {}
        self.exclude_labels  = set(exclude_labels or [])
        self.speaker_category_map_key = speaker_category_map_key

        # Internal CategoryStats shares the same file list and label config
        self.category_stats = CategoryStats(
            files=self.files,
            speaker_annotation=speaker_annotation,
            speaker_category_map_key=speaker_category_map_key,
            category_annotation=category_annotation,
            uri_key=uri_key,
            merge_labels=merge_labels,
            exclude_labels=exclude_labels,
        )

        # Reuse the dicts CategoryStats already built — no second/third iteration
        self._reference  = self.category_stats._speaker   # CategoryStats calls this _speaker
        self._category   = self.category_stats._category  # identical key/value structure ({} if unused)
        # Hypothesis is the only dict that needs its own pass
        self._hypothesis = {f[uri_key]: f[hypothesis_annotation] for f in self.files}

        # Canonical label set comes from CategoryStats (already merged/excluded)
        self.labels = self.category_stats.labels

        # Convenience set for membership tests
        self._category_labels = set(self.labels)

        self._reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_label(self, raw_label):
        """Applies merge/exclude to a raw category label; returns None if excluded."""
        return self.category_stats._resolve_category_label(raw_label)

    def _empty_label_dict(self):
        return dict.fromkeys(self.labels, 0.0)

    def _empty_totals(self):
        return {
            'md':        {'ref':  self._empty_label_dict()},
            'fa':        {'pred': self._empty_label_dict()},
            'confusion': {'ref':  self._empty_label_dict(),
                          'pred': self._empty_label_dict()},
            'correct':   {'ref':  self._empty_label_dict()},
        }

    def _reset(self):
        """Clears computed distributions and the per-file diff/mapping caches."""
        self.error_distribution          = None
        self.overlap_error_distribution  = None
        self.confusion_durations         = None
        self.overlap_confusion_durations = None
        self._diff_cache = {}
        self._mapping_cache = {}

    # ------------------------------------------------------------------
    # Setters — rebuild look-ups and clear stale results
    # ------------------------------------------------------------------

    def set_hypothesis(self, hypothesis_annotation_key):
        self.hypothesis_annotation_key = hypothesis_annotation_key
        self._hypothesis = {f[self.uri_key]: f[hypothesis_annotation_key] for f in self.files}
        self._reset()

    def set_reference(self, speaker_annotation_key):
        self.speaker_annotation_key = speaker_annotation_key
        self._reference = {f[self.uri_key]: f[speaker_annotation_key] for f in self.files}
        self._reset()

    def set_category(self, category_annotation_key):
        self.category_annotation_key = category_annotation_key
        self._category = {f[self.uri_key]: f[category_annotation_key] for f in self.files}
        self._reset()

    def _get_or_compute_mapped_hypothesis(self, key):
        """
        Returns this file's hypothesis Annotation with speaker labels renamed
        to their best-matching reference label, via an optimal one-to-one
        mapping (Hungarian algorithm on total overlap duration).

        This is required because hypotheses will not always match reference
        labels literally, and naming conventions vary across different
        speaker diarization systems. This is unconditional and independent
        of any category (e.g. gender) mapping -- even for files that carry
        their own pre-computed speaker_category_map for categorization,
        hypothesis-to-reference speaker *identity* still needs to be resolved
        here before any error type (md/fa/confusion) can be computed.

        Cached per URI; invalidated by _reset() (called by any set_*()).
        """
        if key not in self._mapping_cache:
            ref_ann = self._reference[key]
            hyp_ann = self._hypothesis[key]
            mapping = HungarianMapper()(hyp_ann, ref_ann)  # {hyp_label: ref_label}
            self._mapping_cache[key] = hyp_ann.rename_labels(mapping=mapping)
        return self._mapping_cache[key]

    def _get_or_compute_diff(self, key):
        """
        Returns the cached IdentificationErrorAnalysis difference annotation
        for a single file URI, computing and storing it on first access.
        Uses the Hungarian-mapped hypothesis for best possible speaker alignment rather than raw label equality.
        """
        if key not in self._diff_cache:
            analyzer = IdentificationErrorAnalysis()
            mapped_hypothesis = self._get_or_compute_mapped_hypothesis(key)
            self._diff_cache[key] = analyzer.difference(
                self._reference[key], mapped_hypothesis
            )
        return self._diff_cache[key]

    def compute_distributions(self, normalized=True, verbose=True, uris=None):
        """
        Single pass over all files that computes and stores:
          - self.error_distribution          (error distribution over all speech)
          - self.overlap_error_distribution  (error distribution over overlap regions only)
          - self.confusion_durations         (confusion durations over all speech)
          - self.overlap_confusion_durations (confusion durations over overlap regions only)

        Always recomputes (no caching)

        Note: this also calls self.category_stats.compute_durations(...),
        which overwrites self.category_stats.durations/.durations_overlap as
        a side effect. If you're using the same CategoryStats instance
        independently elsewhere (e.g. for plot_durations with overlap=False),
        calling this afterwards will overwrite that stored state.
        """
        keys    = list(uris) if uris is not None else list(self._reference.keys())
        mapping = self.category_stats.compute_speaker_category_map(uris=uris)

        all_totals      = self._empty_totals()
        overlap_totals  = self._empty_totals()
        all_conf_raw    = defaultdict(float)
        overlap_conf_raw= defaultdict(float)

        def normalize_conf_label(raw):
            resolved = self._resolve_label(raw)
            if resolved is None or resolved not in self._category_labels:
                return 'UNKNOWN'
            return resolved

        for key in keys:
            ref_ann  = self._reference[key]
            hyp_ann  = self._get_or_compute_mapped_hypothesis(key)
            cat_map  = mapping[key]

            # ── all-speech errors (diff cached per file) ───────────────────────
            errors = self._get_or_compute_diff(key)
            self._accumulate_totals(errors, hyp_ann, ref_ann, cat_map,
                                    all_totals, for_overlap=False, verbose=verbose)
            self._accumulate_conf_raw(errors, cat_map, all_conf_raw,
                                      normalize_conf_label, verbose=verbose)

            # ── overlap-only errors ────────────────────────────────────────────
            overlap_tl = ref_ann.get_overlap()
            if overlap_tl:
                analyzer = IdentificationErrorAnalysis()
                ref_crop  = ref_ann.crop(overlap_tl, mode="intersection")
                hyp_crop  = hyp_ann.crop(overlap_tl, mode="intersection")
                ol_errors = analyzer.difference(ref_crop, hyp_crop)
                self._accumulate_totals(ol_errors, hyp_ann, ref_ann, cat_map,
                                        overlap_totals, for_overlap=True, verbose=verbose)
                self._accumulate_conf_raw(ol_errors, cat_map, overlap_conf_raw,
                                          normalize_conf_label, verbose=verbose)

        # Fetch once — passed into _normalize_totals and reused for confusion normalisation.
        all_dur = self.category_stats.compute_durations(uris=uris, verbose=verbose)

        # Normalize and store error distributions
        self.error_distribution         = self._normalize_totals(all_totals,     normalized, all_dur)
        self.overlap_error_distribution = self._normalize_totals(overlap_totals, normalized, all_dur)

        # Normalize and store confusion durations
        ref_durations = {k: v for k, v in all_dur.items() if k != 'TOTAL'}

        def normalize_conf(raw):
            return {
                pair: dur / ref_durations[pair.split('+')[0]]
                for pair, dur in raw.items()
                if ref_durations.get(pair.split('+')[0], 0.0) > 0
            }

        self.confusion_durations         = normalize_conf(all_conf_raw)
        self.overlap_confusion_durations = normalize_conf(overlap_conf_raw)

    def _accumulate_conf_raw(self, errors, cat_map, conf_raw, normalize_label_fn, verbose=True):
        """Accumulates raw confusion durations from an error annotation."""
        confusion_subset = self.subset_by_error_type(errors, 'confusion')
        for seg, _, speaker in confusion_subset.itertracks(yield_label=True):
            ref_label  = normalize_label_fn(cat_map.get(speaker[1]))
            pred_label = normalize_label_fn(cat_map.get(speaker[2]))
            if ref_label and pred_label:
                conf_raw[ref_label + '+' + pred_label] += seg.duration
            elif verbose and (not ref_label or not pred_label):
                missing = []
                if not ref_label:
                    missing.append(f"ref speaker '{speaker[1]}'")
                if not pred_label:
                    missing.append(f"pred speaker '{speaker[2]}'")
                warnings.warn(
                    f"[_accumulate_conf_raw] Skipping confusion segment: "
                    f"could not resolve category for {', '.join(missing)}.",
                    UserWarning,
                    stacklevel=2,
                )

    def _accumulate_totals(self, errors, hyp_ann, ref_ann, cat_map,
                           totals, for_overlap, verbose):
        """
        Accumulates per-error-type category durations into 'totals' for one file.
        Handles the overlap FA specially (get_overlap_fa) vs standard FA.
        """
        subsets = {
            'md':       self.subset_by_error_type(errors, 'missed detection'),
            'confusion':self.subset_by_error_type(errors, 'confusion'),
            'correct':  self.subset_by_error_type(errors, 'correct'),
        }

        if for_overlap:
            fa_speaker_durations = self.get_overlap_fa(hyp_ann, ref_ann)
            for spk, duration in fa_speaker_durations.items():
                cat = cat_map.get(spk)
                if cat in self._category_labels:
                    totals['fa']['pred'][cat] += duration
        else:
            subsets['fa'] = self.subset_by_error_type(errors, 'false alarm')

        for error_type, subset in subsets.items():
            durations = self.error_category_durations(subset, cat_map, error_type, verbose)
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

    def _normalize_totals(self, totals, normalized, all_dur):
        """Converts raw duration totals into the final error distribution dict.

        all_dur : pre-fetched result of category_stats.compute_durations(),
                  passed in to avoid redundant calls when normalizing multiple
                  totals in one compute cycle.
        """
        def normalize(d, normalized):
            if normalized:
                return {
                    k: v / all_dur[k]
                    for k, v in d.items()
                    if k != 'TOTAL' and all_dur.get(k, 0.0) > 0
                }
            else:
                total = sum(d.values())
                if total == 0:
                    return {k: 0.0 for k in d}
                return {k: v / total for k, v in d.items()}

        return {
            'md':        normalize(totals['md']['ref'],         normalized),
            'fa':        normalize(totals['fa']['pred'],        normalized),
            'confusion': {
                'ref':   normalize(totals['confusion']['ref'],  normalized),
                'pred':  normalize(totals['confusion']['pred'], normalized),
            },
            'correct':   normalize(totals['correct']['ref'],    normalized),
        }

    # ------------------------------------------------------------------
    # Error subsetting
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
                f"Unknown error type '{error_type}'. "
                f"Valid options: {sorted(self._VALID_ERROR_TYPES)}"
            )

        filtered = error_annotation.empty()
        for segment, track, label in error_annotation.itertracks(yield_label=True):
            label_type = label[0] if isinstance(label, tuple) else label
            if label_type.lower() == error_type:
                filtered[segment, track] = label
        return filtered

    # ------------------------------------------------------------------
    # Duration counting
    # ------------------------------------------------------------------

    def error_category_durations(self, error_annotation, category_mapping,
                                 error_type, verbose=True):
        """
        Accumulates speech duration per category label within an error annotation.

        For 'missed detection' / 'correct' : returns {label: duration} by reference speaker
        For 'false alarm'                  : returns {label: duration} by hypothesis speaker
        For 'confusion'                    : returns {'ref': {...}, 'pred': {...}}

        error_annotation  : pyannote Annotation (already filtered to one error type)
        category_mapping  : {speaker_label: category_label} for one file
        error_type        : str
        verbose           : if True, warns about speakers with no category mapping
        """
        if error_annotation is None:
            raise ValueError("error_annotation cannot be None.")

        error_type = error_type.lower()
        error_type = self._ALIASES.get(error_type, error_type)

        if error_type not in self._VALID_ERROR_TYPES:
            raise ValueError(
                f"Unknown error type '{error_type}'. "
                f"Valid options: {sorted(self._VALID_ERROR_TYPES)}"
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
                cat = category_mapping.get(ref_speaker)
                if cat in ref_durations:
                    ref_durations[cat] += seg.duration
                elif ref_speaker is not None:
                    unknown_ref.add(ref_speaker)

            elif error_type == 'false alarm':
                cat = category_mapping.get(pred_speaker)
                if cat in pred_durations:
                    pred_durations[cat] += seg.duration
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
                parts.append(f"reference speakers with no mapping: {sorted(unknown_ref)}")
            if unknown_pred:
                parts.append(f"hypothesis speakers with no mapping: {sorted(unknown_pred)}")
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

    @staticmethod
    def get_overlap_fa(hyp, ref):
        """
        For each predicted overlap region where hyp has more speakers than ref,
        identifies false-alarm speakers and sums their error durations.

        Returns {predicted_speaker_label: total_fa_duration}
        """
        mapper          = HungarianMapper()
        speaker_mapping = mapper(hyp, ref)
        pred_timeline   = hyp.get_overlap()
        fa_durations    = defaultdict(float)

        for segment in pred_timeline:
            pred_speakers = {
                spk for seg, _, spk in hyp.itertracks(yield_label=True) if seg & segment
            }
            ref_speakers = {
                spk for seg, _, spk in ref.itertracks(yield_label=True) if seg & segment
            }

            if len(pred_speakers) <= len(ref_speakers):
                continue

            mapped_pred = {
                p for p, r in speaker_mapping.items()
                if r in ref_speakers and p in pred_speakers
            }
            fa_speakers = pred_speakers - mapped_pred

            for spk in fa_speakers:
                for seg, _, s in hyp.itertracks(yield_label=True):
                    if s == spk:
                        overlap = seg & segment
                        if overlap:
                            fa_durations[spk] += overlap.duration

        return dict(fa_durations)

    # ------------------------------------------------------------------
    # Diff utility
    # ------------------------------------------------------------------

    @staticmethod
    def diff_error_rates(a, b, round_to=None):
        """
        Prints and returns the per-label difference in error rates between
        two error_distribution results (a minus b)
        diff_error_rates(model_a.error_distribution, model_b.error_distribution).
        """
        def diff(x, y):
            d = x - y
            return round(d, round_to) if round_to is not None else d

        def fmt(v):
            sign = '+' if v * 100 > 0 else ''
            return f"{sign}{v * 100:.1f}%"

        def fmt_block(d):
            return "\n".join(f"  * {label}: {fmt(v)}" for label, v in d.items())

        simple_keys = ['md', 'fa', 'correct']
        label_names = {'md': 'Missed Detections', 'fa': 'False Alarms', 'correct': 'Correct'}
        result = {}
        for k in simple_keys:
            result[k] = {label: diff(a[k][label], b[k][label]) for label in a[k]}
        result['confusion'] = {
            'ref':  {l: diff(a['confusion']['ref'][l],  b['confusion']['ref'][l])
                     for l in a['confusion']['ref']},
            'pred': {l: diff(a['confusion']['pred'][l], b['confusion']['pred'][l])
                     for l in a['confusion']['pred']},
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