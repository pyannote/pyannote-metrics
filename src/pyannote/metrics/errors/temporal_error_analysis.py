from pyannote.metrics.errors.identification import IdentificationErrorAnalysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import hashlib
import json

CORRECT_SPEAKER_COLORS = {
    1: "#042C53",
    2: "#185FA5",
    3: "#378ADD",
    4: "#85B7EB",
    5: "#B5D4F4",
}

MD_SPEAKER_COLORS = {
    1: "#7A3D00",
    2: "#C45E00",
    3: "#F77F00",
    4: "#FFB05C",
    5: "#FFD4A8",
}

FA_SPEAKER_COLORS = {
    1: "#7A5A00",
    2: "#C49000",
    3: "#FCBF49",
    4: "#FFD98A",
    5: "#FFECBF",
}

CONFUSION_SPEAKER_COLORS = {
    1: "#5A0000",
    2: "#9A0F0F",
    3: "#D62828",
    4: "#E87070",
    5: "#F5B3B3",
}

ALL_SPEAKER_COLORS = {
    1: "#1A1A2E",
    2: "#3D3D6B",
    3: "#6B6BAE",
    4: "#A3A3D1",
    5: "#D1D1ED",
}

_PALETTE_BY_ERROR_TYPE = {
    "correct":         CORRECT_SPEAKER_COLORS,
    "missed detection": MD_SPEAKER_COLORS,
    "false alarm":     FA_SPEAKER_COLORS,
    "confusion":       CONFUSION_SPEAKER_COLORS,
    None:              ALL_SPEAKER_COLORS,
} 

DEFAULT_LABELS = {
    "x_label": "Signal Value Range",
    "overlay_label": "Average Signal Value",
}

def plot_signal(data, duration):
    """
    Visualize the signal over time

    data: npy array representing signal, not just snr
    duration: float, audio duration in s
    """
    fps = data.shape[0] / duration

    time = np.arange(len(data)) / fps
    
    plt.plot(time, data)
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("Signal")
    plt.tight_layout()
    plt.show()

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

        viz = TemporalErrorAnalysis(
            files=files,
            reference="annotation",
            hypothesis="precision",
            durations="duration",
            signal="snr",
            overlay_signal="clustering_confidence",
            signal_transforms={"clustering_confidence": ["max", "scale_percentage_up"]},
        )

        viz.plot(view="full",
                 show_values=True,
                 show_speaker_breakdown=False,
                 show_overlay=True,
                 title="Error Analysis by SNR",
                 x_label="SNR (dB)",
                 overlay_label="Average Maximum Clustering Confidence",
                 duration_pos="upper right",
                 percentage_pos="upper right",
                 bin_ranges=None)
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
        self._metrics_cache_key = None

    def compute_metrics(self, bin_ranges=None, force=False):
        """
        Set self.metrics and the current cache key.
        Determines whether recomputation is needed, defines bin_ranges and metrics dictionary, and accumulates values across the files.

        bin_ranges: (min, max, interval)
        force: boolean whether to recalculate metrics and override the cache key check
        """
        cache_key = self._compute_metrics_cache_key(bin_ranges)
        
        if not force and self.metrics is not None and cache_key == self._metrics_cache_key:
            print("Using cached metrics. Use force=True to recalculate.")
            return
        
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
        self._metrics_cache_key = cache_key

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

    def plot(self, view="full", show_values=True, show_speaker_breakdown=False,
             show_overlay=False, title=None, x_label=None, overlay_label=None,
             duration_pos="upper right", percentage_pos="upper right", bin_ranges=None, force=False):
        """
        Computes metrics if not already cached, then plots.
        
        Calculates the error analysis with the given bin ranges (or computes them automatically)
        Then plots the duration, percentage, or both error analyses according to the specified view.

        view: "duration" for view according to absolute duration
              "percentage" for view according to percentage relative to each bin
              "full" for both duration and percentage side-by-side
        show_values: boolean for displaying the exact value labels
        show_speaker_breakdown: boolean for displaying the speaker count distribution within correct areas
        show_overlay: boolean for displaying the overlay signal in the percentage plot
        title: user-specified title above the plot
        x_label: user-specified x-axis label, or a default one if None
        overlay_label: user-specified legend label for the overlay signal, or a default one if None
        duration_pos: upper/lower right/left position of the duration plot legend
        percentage_pos: upper/lower right/left position of the percentage plot legend
        bin_ranges: (min,max,interval) tuple of ints specifying the bin sizes and range.
                    Calculates them automatically if None
        force: override the metrics caching check and force a recalculating of the error analysis metrics
        """
        
        self.compute_metrics(bin_ranges=bin_ranges, force=force)

        if x_label is None:
            x_label = DEFAULT_LABELS["x_label"]
        if overlay_label is None:
            overlay_label = DEFAULT_LABELS["overlay_label"]

        if view == "full":
            fig, axes = plt.subplots(1, 2, figsize=(20, 5))
            self._plot_signal_and_errors(axes[0], by_percentage=False, 
                                        show_values=show_values, show_speaker_breakdown=show_speaker_breakdown, show_overlay=show_overlay,
                                        x_label=x_label, overlay_label=overlay_label,
                                        percentage_pos=percentage_pos, duration_pos=duration_pos)
            self._plot_signal_and_errors(axes[1], by_percentage=True,
                                        show_values=show_values, show_speaker_breakdown=show_speaker_breakdown, show_overlay=show_overlay,
                                        x_label=x_label, overlay_label=overlay_label,
                                        percentage_pos=percentage_pos, duration_pos=duration_pos)
            fig.suptitle(title)
            plt.tight_layout()
            plt.show()
        elif view == "duration":
            fig, ax = plt.subplots(figsize=(10, 5))
            self._plot_signal_and_errors(
                ax=ax, by_percentage=False,
                show_values=show_values, show_speaker_breakdown=show_speaker_breakdown,
                show_overlay=show_overlay, x_label=x_label, overlay_label=overlay_label,
                percentage_pos=percentage_pos, duration_pos=duration_pos,
            )
            fig.suptitle(title)
            plt.tight_layout()
            plt.show()
        else:  # view == "percentage"
            fig, ax = plt.subplots(figsize=(10, 5))
            self._plot_signal_and_errors(
                ax=ax, by_percentage=True,
                show_values=show_values, show_speaker_breakdown=show_speaker_breakdown,
                show_overlay=show_overlay, x_label=x_label, overlay_label=overlay_label,
                percentage_pos=percentage_pos, duration_pos=duration_pos,
            )
            fig.suptitle(title)
            plt.tight_layout()
            plt.show()

    def _plot_signal_and_errors(self, ax, by_percentage, show_values=False, show_speaker_breakdown=False, show_overlay=False,
                                 x_label=None, overlay_label=None,
                                 duration_pos="upper right", percentage_pos="upper right"):
        """
        Plots the error analysis
        ax: plot axis for side-by-side comparisons
        show_values: whether to show labels for the values
        show_speaker_breakdown: whether to show the speaker count distribution within correct segments
        show_overlay: whether to show the overlay signal
        x_label: user-defined label for the x-axis
        overlay_label: user-defined label for the overlay signal
        duration_pos: upper/lower right/left position of the legend in the duration plot
        percentage_pos upper/lower right/left position of the legend in the percentage plot
        """
        if show_overlay and self.overlay_signal is None:
            print("show_overlay=True has no effect: no overlay_signal was provided at init.")
            show_overlay = False
            
        metrics_dist = self.metrics
        bins = list(metrics_dist.keys()) # To turn into labels on x-axis
        bin_ranges_split = [tuple(float(x) for x in s.split('_')) for s in bins]

        def get_segments(b):
            entry = metrics_dist[b]
            segs = {}
            for key in ("confusion", "missed detection", "false alarm"):
                segs[key] = entry.get(key, {}).get("duration", 0)
            correct = entry.get("correct", {})
            n_spk_counts = correct.get("n_speakers", {})
            # Filter out 0-speaker seconds
            n_spk_counts = {n: s for n, s in n_spk_counts.items() if n > 0}
            for n, spk_duration in sorted(n_spk_counts.items()):
                bucket = min(n, 5)
                key = f"correct_{bucket}"
                segs[key] = segs.get(key, 0) + spk_duration  # already in seconds, no scaling needed
            return segs

        raw_data = {b: get_segments(b) for b in bins}

        true_max_speakers = max(
            (n for b in metrics_dist
            for n, s in metrics_dist[b].get("correct", {}).get("n_speakers", {}).items()
            if n > 0 and s > 0),
            default=1
        )
        max_bucket = min(true_max_speakers, 5)

        if by_percentage:
            totals = {b: sum(raw_data[b].values()) for b in bins}
            data = {
                b: {k: (v / totals[b] * 100 if totals[b] > 0 else 0)
                    for k, v in raw_data[b].items()}
                for b in bins
            }
        else:
            data = raw_data

        correct_speaker_counts = [n for n in range(1, max_bucket + 1)]

        error_layers = [
            ("confusion",        "#d62828"),
            ("missed detection", "#f77f00"),
            ("false alarm",      "#fcbf49"),
        ]
        text_colors_map = {
            "confusion":        "#ffffff",
            "missed detection": "#ffffff",
            "false alarm":      "#003049",
        }

        if bin_ranges_split is not None:
            bin_widths = np.array([hi - lo + 1 for lo, hi in bin_ranges_split], dtype=float)
            total_range = bin_widths.sum()
            norm_widths = bin_widths / total_range * len(bins)
            bar_centres = np.cumsum(norm_widths) - norm_widths / 2
            tick_labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in bin_ranges_split]
        else:
            norm_widths = np.ones(len(bins))
            bar_centres = np.arange(len(bins), dtype=float)
            tick_labels = bins

        bottoms = np.zeros(len(bins))

        if show_values:
            y_max = max(sum(data[b].values()) for b in bins) or 1
            label_threshold = y_max * 0.05

        if show_speaker_breakdown:
            for n in correct_speaker_counts:
                key = f"correct_{n}"
                label = f"{n}+ spk (correct)" if n == max_bucket == 5 else f"{n} spk (correct)"
                color = self._get_speaker_color(n, "correct")
                vals = np.array([data[b].get(key, 0) for b in bins])
                bars = ax.bar(
                    bar_centres, vals, width=norm_widths * 0.9,
                    label=label, color=color, bottom=bottoms, align="center"
                )
                if show_values:
                    for bar, val, bot in zip(bars, vals, bottoms):
                        if val < label_threshold:
                            continue
                        fmt = f"{val:.1f}{'%' if by_percentage else 's'}"
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bot + val / 2,
                            fmt,
                            ha="center", va="center",
                            fontsize=7, fontweight="bold",
                            color="#ffffff",
                        )
                bottoms += vals
        else:
            # Aggregate all correct buckets into a single bar
            total_correct = np.zeros(len(bins))
            for n in correct_speaker_counts:
                total_correct += np.array([data[b].get(f"correct_{n}", 0) for b in bins])
            bars = ax.bar(
                bar_centres, total_correct, width=norm_widths * 0.9,
                label="Correct", color=self._get_speaker_color(1, "correct"), bottom=bottoms, align="center"
            )
            if show_values:
                for bar, val, bot in zip(bars, total_correct, bottoms):
                    if val < label_threshold:
                        continue
                    fmt = f"{val:.1f}{'%' if by_percentage else 's'}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bot + val / 2,
                        fmt,
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="#ffffff",
                    )
            bottoms += total_correct

        for key, color in error_layers:
            vals = np.array([data[b].get(key, 0) for b in bins])
            bars = ax.bar(
                bar_centres, vals, width=norm_widths * 0.9,
                label=key.title(), color=color, bottom=bottoms, align="center"
            )
            if show_values:
                for bar, val, bot in zip(bars, vals, bottoms):
                    if val < label_threshold:
                        continue
                    fmt = f"{val:.1f}{'%' if by_percentage else 's'}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bot + val / 2,
                        fmt,
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color=text_colors_map[key],
                    )
            bottoms += vals

        ax.yaxis.set_major_locator(plt.MultipleLocator(20) if by_percentage else plt.MultipleLocator(5000))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(10) if by_percentage else plt.MultipleLocator(2500))
        
        ax.grid(axis='y', which='major', linestyle='-', linewidth=0.5, alpha=0.7, zorder=0)
        ax.grid(axis='y', which='minor', linestyle='--', linewidth=0.3, alpha=0.7, zorder=0)

        ax.set_xticks(bar_centres)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Duration (%)" if by_percentage else "Duration (s)")
        ax.set_title("Percentage" if by_percentage else "Absolute Duration")

        handles, labels = ax.get_legend_handles_labels()

        if by_percentage and show_overlay:
            confidence_vals = [float(metrics_dist[b].get("band_mean_overlay", 0)) for b in bins]

            ax.plot(
                bar_centres, confidence_vals,
                color="white", linewidth=1.5, linestyle="--",
                marker="o", markersize=4, label=overlay_label,
                zorder=5,
            )

        if by_percentage:
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(bottom=0)

        handles, labels = ax.get_legend_handles_labels()
        if by_percentage:
            if percentage_pos is not None:
                ax.legend(handles[::-1], labels[::-1], title='Error Type', loc=percentage_pos)
        else:
            if duration_pos is not None:
                ax.legend(handles[::-1], labels[::-1], title='Error Type', loc=duration_pos)

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
    
    def _get_speaker_color(self, n_speakers, error_type=None):
        """
        Gets color for given number of speakers and error type for plotting functions
        n_speakers: int number of speakers active
        error_type: string error type, one of "correct", "missed detection", "false alarm", "confusion" or None for all
        """
        palette = _PALETTE_BY_ERROR_TYPE.get(error_type, ALL_SPEAKER_COLORS)
        return palette.get(n_speakers, palette[5])

    def _compute_metrics_cache_key(self, bin_ranges):
        """
        Compute hash key based on files, signal types, transforms and bin ranges to determine if cached metrics can be reused
        """
        key = {
            "bin_ranges": bin_ranges,
            "signal": self.signal,
            "overlay_signal": self.overlay_signal,
            "signal_transforms": self.signal_transforms,
            "files": [f["uri"] for f in self.files],
        }
        return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()