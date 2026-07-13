"""
Metrics computation for temporal error analysis.

This module bins a time-varying signal (e.g. SNR,
clustering confidence) against diarization/identification error types
(correct, confusion, missed detection, false alarm) and accumulate durations,
overlay-signal means, and speaker-count breakdowns per bin.

See temporal_error_plots.py for visualizing the resulting '.metrics' /
'.bin_ranges'.


---------------------------------------------------------------------------
Example usage
---------------------------------------------------------------------------
from temporal_error_analysis import TemporalErrorAnalysis

One shared 'files' list. Each file dict carries every model's hypothesis
under its own key, e.g.:


files = [
    {
        "uri":       "DH_EVAL_0001",
        "annotation": <pyannote Annotation>,
        "hypothesis": <pyannote Annotation>,
        "duration":  123.4,
        "snr":       np.array([...]),   # shape (n_frames,)
        "clustering_confidence": np.array([...]),  # shape (n_turns, n_speakers)
    },
    ...
]

note: 'files' doesn't change per model, only the 'hypothesis=' key passed
to TemporalErrorAnalysis changes. There only need to be at least one hypothesis and one signal.

--- Basic use: compute and read the metrics dict directly ---

analysis = TemporalErrorAnalysis(
    files=files,
    reference="annotation",
    hypothesis="hypothesis",
    durations="duration",
    signal="snr",
    overlay_signal="clustering_confidence",
    signal_transforms={"clustering_confidence": ["max", "scale_percentage_up"]},
)

# bin_ranges = (-20,80,10) will produce bins from -20 to 80 by steps of 10
# bin_ranges = None will auto-pick nice bin edges from the signal

analysis.compute_metrics(bin_ranges=(-20,80,10))   

analysis.metrics       # {"lo_hi": {"correct": {...}, "confusion": {...}, ...}, ...}
analysis.bin_ranges     # [(lo, hi), (lo, hi), ...] -- same bins as analysis.metrics.keys()

--- Previewing bins before locking them in (e.g. to share across models) ---

preview_bins = analysis.compute_bin_ranges()   # peeks at the signal range, no error accumulation yet
print(preview_bins)                            # inspect before deciding on a fixed (min, max, interval)

SHARED_BINS = (-15, 25, 5)
analysis.compute_metrics(bin_ranges=SHARED_BINS)   # now locked to bins every model can share

--- Extracting plain arrays for anything other than plotting ---

correct_pct = analysis.extract_binned_signal(error_type="correct", metric="percentage")
confusion_seconds = analysis.extract_binned_signal(error_type="confusion", metric="duration")
"""

from pyannote.metrics.errors.identification import IdentificationErrorAnalysis
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm


def interpolate(arr_1, arr_2, audio_duration):
    """
    Interpolates two arrays into the same shape[0] (upsampling to match the longer).
    Supports 1D and 2D arrays. For 2D arrays, shape[1] can differ — only shape[0]
    (the time/frame axis) is interpolated to match.
    Also returns fps and number of frames of the interpolated arrays.
    arr_1 and arr_2: npy arrays representing signals to interpolate
    audio_duration: float audio duration in s
    """
    arr_1 = arr_1.squeeze()
    arr_2 = arr_2.squeeze()

    n_frames_1 = arr_1.shape[0]
    n_frames_2 = arr_2.shape[0]
    fps_1 = n_frames_1 / audio_duration
    fps_2 = n_frames_2 / audio_duration

    def resample(arr, n_from, n_to):
        t_from = np.linspace(0, audio_duration, n_from)
        t_to   = np.linspace(0, audio_duration, n_to)
        if arr.ndim == 1:
            return interp1d(t_from, arr, kind="linear", fill_value="extrapolate")(t_to)
        else:
            # Interpolate each column independently, then stack
            cols = [
                interp1d(t_from, arr[:, i], kind="linear", fill_value="extrapolate")(t_to)
                for i in range(arr.shape[1])
            ]
            return np.stack(cols, axis=1)

    if fps_1 > fps_2:
        arr_2 = resample(arr_2, n_frames_2, n_frames_1)
        fps, n_frames = fps_1, n_frames_1
    elif fps_1 < fps_2:
        arr_1 = resample(arr_1, n_frames_1, n_frames_2)
        fps, n_frames = fps_2, n_frames_2
    else:
        fps, n_frames = fps_1, n_frames_1

    return arr_1, arr_2, fps, n_frames


def get_stair_curve(prediction, scores, target, view_type="entropy", uri=None):
    """
    Reshape a turn-based 2D array and return a frame-based 1D array. Assumes scores corresponds to the turns in the prediction file.

    prediction: annotation object for the prediction file
    scores: npy array (turns, speakers)
    target: (n_frames float, duration float) n_frames comes from the longest length of the signal and overlay_signal
    view_type: "entropy" (overall entropy at a time)
                "max" (score for most likely speaker at a time)
                "difference" (difference between top two scores)
                None (confidence scores as speakers x frames)
    """
    n_frames = target[0]
    duration = target[1]
    n_speakers = scores.shape[1]
    fps = n_frames / duration

    stair_curve = np.zeros((n_frames, n_speakers))

    if len(prediction) != scores.shape[0]:
        raise ValueError(f"Error in {uri}: Unequal number of turns and segments!\n \
                            Prediction segments: {len(prediction)}\n Cluster Turns : {scores.shape[0]}") # most likely wrong prediction file

    # For each turn, get the segment start/end and repeat that turn's corresponding score for the duration of that turn
    # This will stretch out the turn-based scores into a frame-based stair curve.
    for i, (segment, _) in enumerate(prediction.itertracks()):
        start_idx = max(0, int(segment.start * fps))
        end_idx   = min(n_frames, int(segment.end * fps))
        if start_idx >= end_idx:
            end_idx = min(n_frames, start_idx + 1)
        stair_curve[start_idx:end_idx, :] = scores[i, :].reshape(1, -1)

    # The transform type will transform the 2D array into a 1D array of the same length
    if view_type == "entropy":
        stair_curve_clipped = np.clip(stair_curve, 0, None)
        row_sums = stair_curve_clipped.sum(axis=1, keepdims=True)
        probs = np.where(row_sums > 0, stair_curve_clipped / np.where(row_sums > 0, row_sums, 1), 0)

        # entropy is undefined/zero for 1 speaker
        if n_speakers <= 1:
            return np.zeros(n_frames)

        max_entropy = np.log(n_speakers)
        log_probs = np.where(probs > 0, np.log(probs), 0)
        confidence_arr = -np.sum(probs * log_probs, axis=1) / max_entropy
        return confidence_arr

    elif view_type == "max":
        confidence_arr = stair_curve.max(axis=1)
        return confidence_arr

    elif view_type == "difference":
        if stair_curve.shape[1] < 2:
            # Only one speaker so no second max possible
            confidence_arr = np.zeros(stair_curve.shape[0])
        else:
            sorted_scores = np.sort(stair_curve, axis=1)[:, ::-1]
            confidence_arr = sorted_scores[:, 0] - sorted_scores[:, 1]
        return confidence_arr
    else:
        return stair_curve


class TemporalErrorAnalysis:
    """
    Computes, per signal-value bin, the duration/overlay/speaker-count breakdown
    of each DER error type (correct, confusion, missed detection,
    false alarm) across a set of files.

      - compute_metrics(bin_ranges=None) -> populates self.metrics, self.bin_ranges
      - self.metrics: dict keyed by "lo_hi" bin label -> per-error-type dict of
        {"duration", "overlay", "mean_overlay", "n_speakers"} plus a
        "band_mean_overlay" entry per bin.
      - self.bin_ranges: list[(lo, hi)] float tuples, same bins as self.metrics keys.
      - extract_binned_signal(error_type, metric) -> np.array of one value per bin,
        for cumulative-layer comparisons (used by the plotting module's
        plot_comparison, but usable standalone for any custom analysis/export).

    Callers should compute once and reuse
    self.metrics/self.bin_ranges — including from the plotting module, which
    only ever reads these attributes and never triggers a recompute.
    """

    SIGNAL_TRANSFORMS = {
        "entropy":     lambda signal, hyp, target, uri: get_stair_curve(hyp, signal, target, "entropy", uri),
        "max":         lambda signal, hyp, target, uri: get_stair_curve(hyp, signal, target, "max", uri),
        "difference":  lambda signal, hyp, target, uri: get_stair_curve(hyp, signal, target, "difference", uri),
        "squeeze": lambda signal, hyp, target, uri: signal.squeeze(),
        "scale_percentage_up":  lambda signal, hyp, target, uri: signal.squeeze() * 100,
        "scale_percentage_down":  lambda signal, hyp, target, uri: signal.squeeze() / 100,
    }

    def __init__(
        self,
        files: list[dict],
        reference: str,
        hypothesis: str,
        durations: str,
        signal: str,
        overlay_signal: str = None,
        signal_transforms: dict[str, str | list[str]] = None,
    ):
        """
        Initializes the TemporalErrorAnalysis object.

        files: list of dicts, each dict should contain the keys specified by reference, hypothesis, durations, signal, and optionally overlay_signal.
        reference: string key in files dict for reference annotation
        hypothesis: string key in files dict for hypothesis annotation
        durations: string key in files dict for audio duration in seconds
        signal: string key in files dict for time-series signal to analyze (e.g. confidence or SNR)
        overlay_signal: optional string key in files dict for second time series signal
        signal_transforms: optional dict mapping signal keys to either a single transform name or a list of transform names to apply to that signal.
                            Available transforms: "entropy", "max", "difference", "scale_percentage_up", "scale_percentage_down"

        Example
        -------
        files = [
            {
                "uri":       "DH_EVAL_0001",
                "annotation": <pyannote Annotation>,
                "precision": <pyannote Annotation>,
                "duration":  123.4,
                "snr":       np.array([...]),   # shape (n_frames,)
                "clustering_confidence": np.array([...]),  # shape (n_turns, n_speakers)
            },
            ...
        ]

        analysis = TemporalErrorAnalysis(
            files=files,
            reference="annotation",
            hypothesis="precision",
            durations="duration",
            signal="snr",
            overlay_signal="clustering_confidence",
            signal_transforms={"clustering_confidence": ["max", "scale_percentage_up"]},
        )
        analysis.compute_metrics()

        # to visualize these metrics, refer to temporal_error_plots.py
        """
        self.files = files

        self.reference = reference
        self.hypothesis = hypothesis
        self.durations = durations
        self.signal = signal
        self.overlay_signal = overlay_signal
        self.signal_transforms = signal_transforms

        self.metrics = None
        self.bin_ranges = None

    def compute_metrics(self, bin_ranges=None):
        """
        Computes self.metrics and self.bin_ranges by accumulating error/duration/
        speaker-count/overlay values across all files. 
        Call this once and store/reuse the result.

        bin_ranges: (min, max, interval), or None to auto-compute nice bin edges
                    from the signal's value range.
        """
        if bin_ranges is not None:
            min_val = bin_ranges[0]
            max_val = bin_ranges[1]
            interval = bin_ranges[2]
            bin_edges = np.arange(min_val, max_val + interval * 0.5, interval)
            self.bin_ranges = [(float(a), float(b)) for a, b in zip(bin_edges[:-1], bin_edges[1:])]
        else:
            self.bin_ranges = self.compute_bin_ranges()

        durations_dist = {
            f"{lo}_{hi}": {
                error_type: {
                    "duration": 0.0,
                    "overlay": 0.0,
                    "n_speakers": {}              # {n_speakers: seconds, ...}
                }
                for error_type in ["missed detection", "false alarm", "confusion", "correct"]
            }
            for lo, hi in self.bin_ranges
        }

        for file in tqdm(self.files, desc="Processing files"):
            uri = file["uri"]
            ref = file[self.reference]
            hyp = file[self.hypothesis]
            duration = file[self.durations]
            signal = file[self.signal]
            overlay_signal = file[self.overlay_signal] if self.overlay_signal else None

            analyzer = IdentificationErrorAnalysis()
            errors = analyzer.difference(ref, hyp)

            metrics = self._accumulate_file_metrics(signal=signal, overlay_signal=overlay_signal, duration=duration,
                                                    errors=errors, reference=ref, hypothesis=hyp, durations_dist=durations_dist, uri=uri)
        self.metrics = metrics

    def compute_bin_ranges(self):
        """
        Peeks at the concatenation of the files' signal values and calculates nice min/max/interval
        Sets self.bin_ranges as a list of tuples
        Returns the bin ranges
        """
        def nice_number(x, round_=True):
            if x == 0:
                return 0
            exp = np.floor(np.log10(abs(x)))
            f = abs(x) / 10**exp
            thresholds = (1.5, 3, 7) if round_ else (1, 2, 5)
            nice_f = next((n for t, n in zip(thresholds, (1, 2, 5)) if f < t), 10)
            return np.sign(x) * nice_f * 10**exp

        def nice_bin_edges(values, bins=10):
            lo, hi = values.min(), values.max()
            if lo == hi:
                delta = abs(lo) * 0.1 or 0.1
                return np.array([lo - delta, lo + delta])
            interval = nice_number((hi - lo) / bins)
            if interval == 0:
                interval = (hi - lo) / bins
            start = np.floor(lo / interval) * interval
            stop  = np.ceil(hi / interval) * interval
            return np.arange(start, stop + interval * 0.5, interval)

        all_values = np.concatenate([
            self._prepare_signal(
                f[self.signal],
                self.signal,
                hypothesis=f[self.hypothesis],
                target=(f[self.signal].shape[0], f[self.durations]),
                uri=f["uri"],
            )
            for f in self.files
        ])
        print(f"all_values range: {all_values.min():.6f} – {all_values.max():.6f}")
        bin_edges = nice_bin_edges(all_values, bins=10)
        self.bin_ranges = [(float(a), float(b)) for a, b in zip(bin_edges[:-1], bin_edges[1:])]
        return self.bin_ranges

    def extract_binned_signal(self, error_type="correct", metric="percentage"):
        """
        Returns a 1D np.array with one value per bin (in self.metrics' bin order),
        for the cumulative set of error layers up to and including 'error_type'
        (using the fixed layer order: correct, confusion, missed detection, false alarm).

        error_type: "correct", "confusion", "missed detection", or "false alarm"
                    This is the topmost layer to include in the cumulative sum.
        metric: "percentage" (cumulative duration / total duration * 100),
                "duration" (cumulative duration in seconds),
                or "mean_overlay" (mean_overlay of error_type's own bucket, not cumulative).
        """
        LAYER_ORDER = ["correct", "confusion", "missed detection", "false alarm"]

        if self.metrics is None:
            raise RuntimeError("Call compute_metrics() first.")

        if error_type == "false alarm":
            layers_below = LAYER_ORDER[:LAYER_ORDER.index(error_type)]
        else:
            layers_below = LAYER_ORDER[:LAYER_ORDER.index(error_type) + 1]

        out = []
        for bin_label, bin_data in self.metrics.items():
            if metric == "percentage":
                total = sum(
                    b.get("duration", 0)
                    for b in bin_data.values()
                    if isinstance(b, dict)
                )
                dur = sum(bin_data.get(layer, {}).get("duration", 0) for layer in layers_below)
                val = (dur / total * 100) if total > 0 else 0.0
            elif metric == "duration":
                val = sum(bin_data.get(layer, {}).get("duration", 0) for layer in layers_below)
            elif metric == "mean_overlay":
                val = bin_data.get(error_type, {}).get("mean_overlay", 0.0)
            else:
                raise ValueError(f"Unknown metric '{metric}'. Choose: percentage, duration, mean_overlay")
            out.append(val)

        return np.array(out)

    def _accumulate_file_metrics(self, signal, overlay_signal, duration, errors, reference, hypothesis, durations_dist, uri=None):
        """
        First prepare the signals by making them the same shape and applying any transforms.
        For each SNR bin, calculate the duration of the signal, overlay signal, and each error type.
        Within each error type within each bin, calculate:
        - The duration that error is active
        - The duration of the overlay signal
        - The number of speakers active and the respective durations when that number is active
        This works on one file at a time and accumulates the values in durations_dist.
        Returns the metrics dictionary thus far.
        """

        if overlay_signal is not None:
            _, _, fps, n_frames = interpolate(overlay_signal, signal, duration)
            signal = self._prepare_signal(signal, self.signal, hypothesis, (n_frames,duration), uri)
            overlay_signal = self._prepare_signal(overlay_signal, self.overlay_signal, hypothesis, (n_frames,duration), uri)
            if overlay_signal.ndim > 1:
                raise ValueError(f"Overlay signal array from {uri} has unexpected shape: {overlay_signal.shape}. Try applying a transform to reduce it to 1D.")
        else:
            signal = self._prepare_signal(signal, self.signal, hypothesis, (signal.shape[0],duration), uri)
            signal = signal.squeeze()
            fps = signal.shape[0] / duration

        seconds_per_frame = 1.0 / fps

        speaker_counts = reference.discretize(resolution=seconds_per_frame, duration=duration).data.sum(axis=1)
        frame_error_types = errors.discretize(resolution=seconds_per_frame, duration=duration)

        error_matrix = frame_error_types.data
        error_labels = [label[0] for label in frame_error_types.labels]    # (n_labels,)

        error_masks = {}
        for error_type in ["confusion", "missed detection", "false alarm", "correct"]:
            if error_type not in error_labels:
                continue

            # Get all columns corresponding to this error type
            col_indices = [i for i, label in enumerate(error_labels) if label == error_type]
            if not col_indices:
                continue
            # True wherever ANY of those columns is active
            error_mask = error_matrix[:, col_indices].any(axis=1)
            error_masks.update({error_type: error_mask})

        for i, (lo, hi) in enumerate(self.bin_ranges): # O(10)
            bin_label = f"{lo}_{hi}"

            if i == len(self.bin_ranges) - 1:
                bin_mask = (signal >= lo) & (signal <= hi) # include right edge for last bin
            else:
                bin_mask = (signal >= lo) & (signal < hi) # (n_frames,)

            for error_type, error_mask in error_masks.items(): # O(4)
                active_mask = bin_mask & error_mask  # (n_frames,)

                if not active_mask.any():
                    continue

                bucket = durations_dist[bin_label][error_type]

                bucket["duration"] += active_mask.sum() * seconds_per_frame

                if overlay_signal is not None:
                    overlay_vals = overlay_signal[active_mask]
                    overlay_valid = ~np.isnan(overlay_vals)
                    bucket["overlay"] += overlay_vals[overlay_valid].sum() * seconds_per_frame

                active_speaker_counts = speaker_counts[active_mask].astype(int)
                unique_spk, counts = np.unique(active_speaker_counts, return_counts=True)
                for spk, count in zip(unique_spk, counts):
                    spk = int(spk)
                    bucket["n_speakers"][spk] = bucket["n_speakers"].get(spk, 0) + count * seconds_per_frame

        durations_dist = self._mean_overlay_per_bin(durations_dist)
        return durations_dist

    @staticmethod
    def _mean_overlay_per_bin(metrics):
        """
        Calculate the mean overlay value for each bin for a given metrics distribution.
        """
        for bin_label in metrics:
            total_confidence = 0.0
            total_confidence_duration = 0.0

            for error_type, bucket in metrics[bin_label].items():
                if not isinstance(bucket, dict):
                    continue
                duration = bucket["duration"]
                if duration > 0:
                    bucket["mean_overlay"] = bucket["overlay"] / duration
                total_confidence += bucket["overlay"]
                total_confidence_duration += duration

            if total_confidence_duration > 0:
                metrics[bin_label]["band_mean_overlay"] = (
                    total_confidence / total_confidence_duration
                )

        return metrics

    def _prepare_signal(self, signal, field_name, hypothesis, target, uri):
        """
        Applies every specified transform to the given signal and tells the user how to fix transform errors.
        Returns the transformed signal

        signal: signal to transform
        field_name: name of the signal, for use in printing messages to the user
        hypothesis: annotations with segment timestamps to transform turn based signals into frame based signals
        target: (n_frames, duration) tuple of floats for the signal's target transformation
        uri: uri of the current file
        """
        transform_spec = self.signal_transforms.get(field_name) if self.signal_transforms else None

        if signal.ndim == 1 and transform_spec is None:
            return signal  # nothing to do

        if signal.ndim > 1 and transform_spec is None:
            raise ValueError(
                f"Signal '{field_name}' is 2D but no transform was registered for it. "
                f"Pass signal_transforms={{'{field_name}': 'transform_name'}} where "
                f"transform_name is one of: {list(self.SIGNAL_TRANSFORMS.keys())}"
            )

        # Normalize to a list so single strings and lists are handled uniformly
        transforms = [transform_spec] if isinstance(transform_spec, str) else list(transform_spec)

        for transform_name in transforms:
            transform = self.SIGNAL_TRANSFORMS.get(transform_name)
            if transform is None:
                raise ValueError(
                    f"Unknown transform '{transform_name}' for signal '{field_name}'. "
                    f"Available: {list(self.SIGNAL_TRANSFORMS.keys())}"
                )
            signal = transform(signal, hyp=hypothesis, target=target, uri=uri)

        return signal