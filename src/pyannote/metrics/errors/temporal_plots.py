"""
Visualization for TemporalErrorAnalysis.

This file create visualizations for an already-computed
TemporalErrorAnalysis: it reads '.metrics' / '.bin_ranges' (and calls the
public 'extract_binned_signal()' metrics API) but never calls
'compute_metrics()' itself. Callers are expected to call
'analysis.compute_metrics()' once, then customize the plot as many times as they like.

---------------------------------------------------------------------------
Example usage
---------------------------------------------------------------------------

from temporal_error_analysis import TemporalErrorAnalysis

One shared 'files' list. Each file dict carries every model's hypothesis
under its own key, e.g.:
  {
      "uri": "DH_EVAL_0001",
      "annotation": <reference Annotation>,
      "duration": 123.4,
      "snr": np.array([...]),
      "model_a_prediction": <hypothesis Annotation>,
      "model_b_prediction":   <hypothesis Annotation>,
      "model_c_prediction":     <hypothesis Annotation>,
      "model_d_prediction":     <hypothesis Annotation>,
      ...
  }
note: 'files' doesn't change per model, only the 'hypothesis=' key passed
to TemporalErrorAnalysis changes. There only need to be at least one hypothesis and one signal.

--- plot_signal: sanity-check a raw signal before binning anything ---

plot_signal(files[0]["snr"], files[0]["duration"])

--- plot: one model's error breakdown, binned by the signal ---

model_a = TemporalErrorAnalysis(files=files, reference="annotation",
                                     hypothesis="model_a_prediction", durations="duration", signal="snr")
model_a.compute_metrics()   # bin_ranges=None will auto-pick nice bin edges

plot(model_a, view="full", show_values=True, show_overlay=False,
     x_label="SNR (dB)", title="Model A error breakdown by SNR")

--- plot_comparison: compare error analysis for exactly two models, with clearly illustrated per-bin deltas ---

model_b = TemporalErrorAnalysis(files=files, reference="annotation",
                                  hypothesis="model_b_prediction", durations="duration", signal="snr")
model_b.compute_metrics(bin_ranges=model_a.bin_ranges)   # reuse model_a's bins so they line up

plot_comparison(model_a, model_b, error_type="correct", metric="percentage",
                 model_label="Model A", other_label="Model B", x_label="SNR (dB)")

--- plot_comparison_multi: many models at once vs. one baseline ---
    Every instance -- baseline included -- must share identical bins, so
    compute bin_ranges once and reuse it everywhere.

SHARED_BINS = (-15, 25, 5)   # (min, max, interval) -- pick once, reuse everywhere
MODEL_KEYS = ["-10dB", "0dB", "10dB", "-10_0"]

# model a serves as the baseline
model_a = TemporalErrorAnalysis(files=files, reference="annotation",
                                     hypothesis="model_a_prediction", durations="duration", signal="snr")
model_a.compute_metrics(bin_ranges=SHARED_BINS)

others = {}
for key in MODEL_KEYS:
    temp = TemporalErrorAnalysis(files=files, reference="annotation",
                                 hypothesis=key, durations="duration", signal="snr")
    temp.compute_metrics(bin_ranges=SHARED_BINS)
    others[key] = temp

plot_comparison_multi(model_a, others, error_type="correct", metric="percentage",
                       baseline_label="control", x_label="SNR (dB)")
"""


import numpy as np
import matplotlib.pyplot as plt

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
    Visualize a signal over time.
    Helpful for sanity checking inputs before passing to TemporalErrorAnalysis.

    data: npy array representing a signal
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


def _get_speaker_color(n_speakers, error_type=None):
    """
    Gets color for given number of speakers and error type.
    n_speakers: int number of speakers active
    error_type: string error type, one of "correct", "missed detection", "false alarm", "confusion" or None for all
    """
    palette = _PALETTE_BY_ERROR_TYPE.get(error_type, ALL_SPEAKER_COLORS)
    return palette.get(n_speakers, palette[5])


def _compute_bar_geometry(bins):
    """
    Computes the x-axis bar layout shared by plot() and plot_comparison():
    bin-width-proportional bar centres, normalized widths, and human-readable
    tick labels.

    bins: list of "lo_hi" bin-label strings (e.g. analysis.metrics.keys()).
    Returns (bar_centres, norm_widths, tick_labels).
    """
    bin_ranges_split = [tuple(float(x) for x in s.split('_')) for s in bins]
    bin_widths = np.array([hi - lo + 1 for lo, hi in bin_ranges_split], dtype=float)
    total_range = bin_widths.sum()
    norm_widths = bin_widths / total_range * len(bins)
    bar_centres = np.cumsum(norm_widths) - norm_widths / 2
    tick_labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in bin_ranges_split]
    return bar_centres, norm_widths, tick_labels


def _require_metrics(analysis):
    if analysis.metrics is None:
        raise RuntimeError(
            "analysis.compute_metrics() must be called before plotting"
        )


def _plot_signal_and_errors(analysis, ax, by_percentage, show_values=False, show_speaker_breakdown=False, show_overlay=False,
                             x_label=None, overlay_label=None,
                             duration_pos="upper right", percentage_pos="upper right"):
    """
    Plots the error analysis for one TemporalErrorAnalysis instance onto one axis.

    analysis: a TemporalErrorAnalysis with compute_metrics() already called.
    ax: plot axis for side-by-side comparisons
    show_values: whether to show labels for the values
    show_speaker_breakdown: whether to show the speaker count distribution within correct segments
    show_overlay: whether to show the overlay signal
    x_label: user-defined label for the x-axis
    overlay_label: user-defined label for the overlay signal
    duration_pos: upper/lower right/left position of the legend in the duration plot
    percentage_pos: upper/lower right/left position of the legend in the percentage plot
    """
    _require_metrics(analysis)

    if show_overlay and analysis.overlay_signal is None:
        print("show_overlay=True has no effect: no overlay_signal was provided at init.")
        show_overlay = False

    metrics_dist = analysis.metrics
    bins = list(metrics_dist.keys()) # To turn into labels on x-axis
    bar_centres, norm_widths, tick_labels = _compute_bar_geometry(bins)

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

    bottoms = np.zeros(len(bins))

    if show_values:
        y_max = max(sum(data[b].values()) for b in bins) or 1
        label_threshold = y_max * 0.05

    if show_speaker_breakdown:
        for n in correct_speaker_counts:
            key = f"correct_{n}"
            label = f"{n}+ spk (correct)" if n == max_bucket == 5 else f"{n} spk (correct)"
            color = _get_speaker_color(n, "correct")
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
            label="Correct", color=_get_speaker_color(1, "correct"), bottom=bottoms, align="center"
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


def plot(analysis, view="full", show_values=True, show_speaker_breakdown=False,
         show_overlay=False, title=None, x_label=None, overlay_label=None,
         duration_pos="upper right", percentage_pos="upper right"):
    """
    Plots the duration, percentage, or both error analyses for 'analysis',
    according to the specified view. Requires call to analysis.compute_metrics() beforehand.

    analysis: a TemporalErrorAnalysis with compute_metrics() already called.
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
    """
    _require_metrics(analysis)

    if x_label is None:
        x_label = DEFAULT_LABELS["x_label"]
    if overlay_label is None:
        overlay_label = DEFAULT_LABELS["overlay_label"]

    if view == "full":
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        _plot_signal_and_errors(analysis, axes[0], by_percentage=False,
                                    show_values=show_values, show_speaker_breakdown=show_speaker_breakdown, show_overlay=show_overlay,
                                    x_label=x_label, overlay_label=overlay_label,
                                    percentage_pos=percentage_pos, duration_pos=duration_pos)
        _plot_signal_and_errors(analysis, axes[1], by_percentage=True,
                                    show_values=show_values, show_speaker_breakdown=show_speaker_breakdown, show_overlay=show_overlay,
                                    x_label=x_label, overlay_label=overlay_label,
                                    percentage_pos=percentage_pos, duration_pos=duration_pos)
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    elif view == "duration":
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_signal_and_errors(
            analysis, ax=ax, by_percentage=False,
            show_values=show_values, show_speaker_breakdown=show_speaker_breakdown,
            show_overlay=show_overlay, x_label=x_label, overlay_label=overlay_label,
            percentage_pos=percentage_pos, duration_pos=duration_pos,
        )
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    else:  # view == "percentage"
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_signal_and_errors(
            analysis, ax=ax, by_percentage=True,
            show_values=show_values, show_speaker_breakdown=show_speaker_breakdown,
            show_overlay=show_overlay, x_label=x_label, overlay_label=overlay_label,
            percentage_pos=percentage_pos, duration_pos=duration_pos,
        )
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()


def plot_comparison(analysis, other, error_type="correct", metric="percentage",
                     model_label="Model A", other_label="Model B",
                     show_values=False, show_speaker_breakdown=False,
                     title=None, x_label=None,
                     duration_pos="upper right", percentage_pos="upper right"):
    """
    Compare 'analysis' (the base model, drawn as the stacked bar chart) against
    another TemporalErrorAnalysis instance, drawing both models' curves on top
    of the bars and shading the gap between them: green where 'other' is
    better than 'analysis', red where it's worse. Each bin is labeled with the
    delta between the two models.

    analysis: base TemporalErrorAnalysis, with compute_metrics() already called.
    other: another TemporalErrorAnalysis instance. Must already have
           compute_metrics() called with bin_ranges identical to analysis's
           (same bin edges/labels), so the two curves line up bin-for-bin.
    error_type: which cumulative error layer to compare -- "correct", "confusion",
                "missed detection", or "false alarm". Passed to extract_binned_signal.
    metric: "percentage", "duration", or "mean_overlay" -- passed to extract_binned_signal.
            "percentage" (the default) is almost always what you want here, since it's
            directly comparable across bins with very different total durations.
    model_label / other_label: legend labels for analysis's curve / other's curve.
    show_values: whether to also show the per-segment duration/percentage labels on the
                 underlying bars (off by default here, since the per-bin delta labels
                 already add text to the same area).
    show_speaker_breakdown: whether to break the "Correct" bar segment down by active
                             speaker count.
    title: user-specified title above the plot. Defaults to a description of what's
           being compared.
    x_label: user-specified x-axis label, or a default one if None.
    duration_pos / percentage_pos: kept for signature consistency with plot(); only
                                    percentage_pos affects this plot's legend, since
                                    plot_comparison only draws the percentage view.
    """
    _require_metrics(analysis)
    _require_metrics(other)

    if list(analysis.metrics.keys()) != list(other.metrics.keys()):
        raise ValueError(
            "analysis and other must share identical bins to be compared bin-for-bin. "
            "Call compute_metrics(bin_ranges=...) with the same bin_ranges on both "
            "instances (e.g. compute bin_ranges once on analysis, then pass that tuple "
            "to other.compute_metrics())."
        )

    self_values = analysis.extract_binned_signal(error_type=error_type, metric=metric)
    other_values = other.extract_binned_signal(error_type=error_type, metric=metric)

    if x_label is None:
        x_label = DEFAULT_LABELS["x_label"]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Base chart: analysis's stacked percentage bars, with no overlay line of its own --
    # the two model curves below are drawn manually so we can shade between them.
    _plot_signal_and_errors(
        analysis, ax=ax, by_percentage=True,
        show_values=show_values, show_speaker_breakdown=show_speaker_breakdown,
        show_overlay=False, x_label=x_label, overlay_label=None,
        duration_pos=duration_pos, percentage_pos=percentage_pos,
    )

    bins = list(analysis.metrics.keys())
    bar_centres, norm_widths, tick_labels = _compute_bar_geometry(bins)

    ax.plot(bar_centres, self_values, color="white", linewidth=1.8, linestyle="-",
            marker="o", markersize=4, label=model_label, zorder=6)
    ax.plot(bar_centres, other_values, color="black", linewidth=1.8, linestyle="--",
            marker="s", markersize=4, label=other_label, zorder=6)

    better_mask = other_values >= self_values
    ax.fill_between(bar_centres, self_values, other_values, where=better_mask,
                     color="#2a9d2a", alpha=0.35, interpolate=True, zorder=4,
                     label=f"{other_label} better")
    ax.fill_between(bar_centres, self_values, other_values, where=~better_mask,
                     color="#d62828", alpha=0.35, interpolate=True, zorder=4,
                     label=f"{other_label} worse")

    unit = "pp" if metric == "percentage" else ("s" if metric == "duration" else "")
    for x, a, b in zip(bar_centres, self_values, other_values):
        delta = b - a
        if abs(delta) < 1e-9:
            continue
        sign = "+" if delta > 0 else ""
        ax.annotate(
            f"{sign}{delta:.1f}{unit}",
            xy=(x, (a + b) / 2),
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="#1a6b1a" if delta > 0 else "#a31e1e",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
            zorder=7,
        )

    # De-duplicate legend entries (the bar legend plus our two lines/two fills)
    handles, labels = ax.get_legend_handles_labels()
    seen, uniq_handles, uniq_labels = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_handles.append(h)
        uniq_labels.append(l)
    ax.legend(uniq_handles[::-1], uniq_labels[::-1], title="Comparison", loc=percentage_pos)

    fig.suptitle(title or f"{model_label} vs {other_label}: {error_type.title()} ({metric})")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Comparing many models against a baseline at once
# ---------------------------------------------------------------------------
# Unlike plot_comparison() above (built for exactly two models, with inline
# delta labels and fill_between shading), these are for larger sweeps
# e.g. 10+ models trained at different augmentation SNRs
# 

LAYER_ORDER = ["correct", "confusion", "missed detection", "false alarm"]


def _bin_totals(instance):
    """Total duration (all error types) accumulated in each bin -- used to
    detect bins with little or no data, so they can be excluded rather than
    silently plotted as 0."""
    return np.array([
        sum(b.get("duration", 0) for b in bin_data.values() if isinstance(b, dict))
        for bin_data in instance.metrics.values()
    ])


def _extract_layer_value(instance, error_type, metric="percentage", min_duration=0.0):
    """
    Per-bin value for exactly 'error_type''s own duration -- NOT the
    cumulative/stacked value that instance.extract_binned_signal() returns.

    This works around a quirk in extract_binned_signal(): it's designed to
    trace the top edge of a given layer's band in a stacked bar chart, which
    means for most error types it returns a cumulative sum (e.g. "confusion"
    returns correct+confusion). For "false alarm" specifically it's special-
    cased to return the cumulative sum THROUGH "missed detection" instead
    (since the true cumulative-through-false-alarm would trivially be ~100%),
    which makes it numerically identical to what "missed detection" returns --
    i.e. two different error_type requests produce the same curve.

    For comparing error rates across models (rather than aligning a line to a
    stacked bar chart), each layer's own value is what you actually want, so
    this bypasses extract_binned_signal's cumulative logic entirely.
    """
    totals = _bin_totals(instance)
    vals = np.array([
        bin_data.get(error_type, {}).get("duration", 0.0)
        for bin_data in instance.metrics.values()
    ], dtype=float)
    if metric == "percentage":
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.where(totals > 0, vals / totals * 100, 0.0)
    elif metric != "duration":
        raise ValueError(f"metric must be 'percentage' or 'duration', got {metric!r}")
    vals[totals <= min_duration] = np.nan
    return vals


def _extract_cumulative_value(instance, error_type, metric="percentage", min_duration=0.0):
    """
    Per-bin cumulative value through 'error_type' in stack order (correct ->
    confusion -> missed detection -> false alarm) -- i.e. the height of the
    top edge of that layer's band, if this model's own data were stacked the
    same way the baseline's bar chart is. This is what makes an overlaid line
    land at the same vertical position as the corresponding colored band,
    instead of a raw own-value line which for confusion/missed detection/
    false alarm sits near the bottom of the axis (since those layers are
    individually small) regardless of where that band actually is in the
    stack.

    Deliberately does NOT reuse instance.extract_binned_signal(), which
    special-cases "false alarm" to alias onto "missed detection"'s cumulative
    value -- see _extract_layer_value's docstring. Here, cumulative-through-
    false-alarm is correctly the top of the whole stack -- always 100% for
    metric="percentage", since everything sums to 100% by definition. That's
    intentional: it's a flat reference line at the top of the chart, which is
    in fact where the top of the false-alarm band always sits.
    """
    idx = LAYER_ORDER.index(error_type)
    layers = LAYER_ORDER[: idx + 1]
    totals = _bin_totals(instance)
    cum_seconds = np.zeros(len(totals))
    for layer in layers:
        cum_seconds += np.array([
            bin_data.get(layer, {}).get("duration", 0.0)
            for bin_data in instance.metrics.values()
        ])
    if metric == "percentage":
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.where(totals > 0, cum_seconds / totals * 100, 0.0)
    elif metric == "duration":
        vals = cum_seconds
    else:
        raise ValueError(f"metric must be 'percentage' or 'duration', got {metric!r}")
    vals[totals <= min_duration] = np.nan
    return vals


def _extract_with_gaps(instance, error_type, metric, min_duration=0.0, cumulative=None):
    """
    Bin values for 'error_type', with bins below 'min_duration' seconds of
    total data set to NaN instead of 0 (avoids fake plunges/spikes at sparse
    bins).

    cumulative=None (default, "auto"): uses each layer's own value for
    error_type="correct" (already the natural top-of-stack position, since
    correct is the base layer), and the corrected cumulative/stack-position
    value (see _extract_cumulative_value) for every other error type, so the
    overlaid line lands where that colored band actually is in the chart.

    cumulative=False forces the own-value extraction for any error_type.
    cumulative=True forces the cumulative/stack-position extraction for any
    error_type (redundant with "correct", since they're identical there).
    """
    if cumulative is None:
        cumulative = (error_type != "correct")
    if cumulative:
        return _extract_cumulative_value(instance, error_type, metric=metric, min_duration=min_duration)
    return _extract_layer_value(instance, error_type, metric=metric, min_duration=min_duration)


def _apply_dense_grid(ax, ylim=None, major_step=None, minor_step=None, default_major=20):
    """
    Sets denser, nicer-spaced y-gridlines than the default fixed
    MultipleLocator(20) used for percentage plots elsewhere -- fine for a
    0-100 view, much too coarse once you've zoomed into e.g. (80, 100).

    If major_step/minor_step aren't given, picks a "nice" major step (1, 2,
    2.5, 5, or 10 x a power of ten) that gives roughly 8-10 major gridlines
    across the current y-range, with minor gridlines at a quarter of that.
    """
    if major_step is None:
        span = (ylim[1] - ylim[0]) if ylim is not None else default_major * 5
        raw = span / 8
        magnitude = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
        for m in (1, 2, 2.5, 5, 10):
            if raw <= m * magnitude:
                major_step = m * magnitude
                break
        else:
            major_step = 10 * magnitude
    if minor_step is None:
        minor_step = major_step / 4

    ax.yaxis.set_major_locator(plt.MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_step))
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.6, alpha=0.7, zorder=0)
    ax.grid(axis="y", which="minor", linestyle="--", linewidth=0.35, alpha=0.45, zorder=0)


def plot_comparison_multi(
    baseline,
    others: dict,
    error_type: str = "correct",
    metric: str = "percentage",
    baseline_label: str = "Control",
    title: str = None,
    x_label: str = None,
    show_values: bool = False,
    show_speaker_breakdown: bool = False,
    cmap_name: str = "tab20",
    legend_ncol: int = 1,
    figsize=(15, 7),
    ylim: tuple = None,
    min_duration: float = 0.0,
    add_delta_panel: bool = True,
    delta_ylim: tuple = None,
    major_gridline_step: float = None,
    minor_gridline_step: float = None,
    delta_major_gridline_step: float = None,
    delta_minor_gridline_step: float = None,
    cumulative: bool = None,
):
    """
    Compare 'baseline' (a TemporalErrorAnalysis instance, drawn as the stacked
    percentage bar chart) against many other TemporalErrorAnalysis instances at
    once, each drawn as its own line, with one combined legend placed outside
    the axes (essential once you have 13+ models -- inline delta labels and
    fill_between shading, as in plot_comparison, become unreadable at that
    count).

    baseline: TemporalErrorAnalysis instance for the control model. Must
              already have compute_metrics() called.
    others: dict mapping {model_label: TemporalErrorAnalysis instance}. Every
            instance -- baseline included -- MUST have been built with
            compute_metrics(bin_ranges=<same (min, max, interval) tuple>) so
            all curves line up bin-for-bin. Don't rely on automatic bin
            detection here, since that's computed per-instance from that
            model's own signal values and can silently drift between models.
    error_type / metric: forwarded to extract_binned_signal on every instance
                          (same meaning as in plot_comparison).
    baseline_label: legend label for the baseline curve.
    show_values / show_speaker_breakdown: forwarded to the underlying bar plot
                                           for the baseline.
    cmap_name: matplotlib colormap used to assign each "other" model a
               distinct line color.
    legend_ncol: number of columns in the outside legend, useful if you have
                 many models and want a more compact legend block.
    ylim: optional (lo, hi) tuple to zoom the main y-axis. With 13 models
          clustered a few points apart near the top of the stack, something
          like (70, 100) usually separates the curves far better than the
          full 0-100 range.
    min_duration: bins in 'baseline' with less than this many seconds of total
                  data are treated as empty (NaN) rather than plotted -- this
                  removes the fake plunges/spikes at sparsely-populated edge
                  bins. Try inspecting _bin_totals(baseline) first to pick a
                  sensible threshold for your data.
    add_delta_panel: if True, adds a second panel below the main plot showing
                      (other - baseline) directly for every model, per bin.
                      This is usually the more informative view once curves
                      are this close together -- it turns "can you spot the
                      tiny gap between these lines" into a chart where the
                      differences are the y-axis.
    delta_ylim: optional (lo, hi) tuple to zoom the delta panel's y-axis.
    major_gridline_step / minor_gridline_step: override the auto-picked
          gridline spacing on the main panel. Leave as None to auto-pick a
          "nice" step (~8-10 major gridlines) based on 'ylim', which is
          usually what you want once you've zoomed in -- the default fixed
          20-point gridlines are too coarse for a zoomed range like (80, 100).
    delta_major_gridline_step / delta_minor_gridline_step: same, for the
          delta panel.
    cumulative: None (default, "auto") plots error_type="correct" as its own
          value (already correctly positioned, since correct is the base
          layer) and every other error_type as the corrected cumulative
          stack-position value, so the line lands at the same height as that
          error type's colored band in the baseline's bar chart. Pass False
          to force each layer's own (non-cumulative) value regardless of
          error_type, or True to force the cumulative/stack-position value
          for every error_type including "correct" (identical to the auto
          behavior for "correct", since it's the base layer either way).
    """
    if baseline.metrics is None:
        raise RuntimeError("Call compute_metrics() on baseline first.")
    for label, other in others.items():
        if other.metrics is None:
            raise RuntimeError(f"Call compute_metrics() on '{label}' first.")
        if list(baseline.metrics.keys()) != list(other.metrics.keys()):
            raise ValueError(
                f"Bin mismatch between baseline and '{label}'. Make sure both "
                f"were computed with compute_metrics(bin_ranges=<same tuple>)."
            )

    if x_label is None:
        x_label = DEFAULT_LABELS["x_label"]

    if add_delta_panel:
        fig, (ax, dax) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08},
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        dax = None

    # Draw the baseline's stacked bars, but suppress its own legend -- we'll
    # build a single combined legend for bars + all model lines afterwards.
    _plot_signal_and_errors(
        baseline,
        ax=ax,
        by_percentage=True,
        show_values=show_values,
        show_speaker_breakdown=show_speaker_breakdown,
        show_overlay=False,
        x_label=x_label if dax is None else None,
        overlay_label=None,
        duration_pos=None,
        percentage_pos=None,
    )
    bar_handles, bar_labels = ax.get_legend_handles_labels()

    bins = list(baseline.metrics.keys())
    bar_centres, norm_widths, tick_labels = _compute_bar_geometry(bins)
    baseline_values = _extract_with_gaps(baseline, error_type, metric, min_duration, cumulative=cumulative)

    baseline_line, = ax.plot(
        bar_centres, baseline_values,
        color="white", linewidth=2.4, linestyle="-",
        marker="o", markersize=5, zorder=6, label=baseline_label,
    )

    cmap = plt.get_cmap(cmap_name, max(len(others), 1))
    unit = "pp" if metric == "percentage" else ("s" if metric == "duration" else "")
    model_lines = [baseline_line]
    avg_deltas = {}
    for i, (label, other) in enumerate(others.items()):
        other_values = _extract_with_gaps(other, error_type, metric, min_duration, cumulative=cumulative)
        avg_delta = np.nanmean(other_values - baseline_values)
        avg_deltas[label] = avg_delta
        sign = "+" if avg_delta >= 0 else ""
        color = cmap(i)
        line, = ax.plot(
            bar_centres, other_values,
            color=color, linewidth=1.6, marker="s", markersize=3.5,
            alpha=0.9, zorder=5, label=f"{label} ({sign}{avg_delta:.1f}{unit} avg)",
        )
        model_lines.append(line)

        if dax is not None:
            delta = other_values - baseline_values
            dax.plot(bar_centres, delta, color=color, linewidth=1.6,
                      marker="s", markersize=3.5, alpha=0.9, zorder=5)

    # "correct" is better when higher; error-type layers (confusion, missed
    # detection, false alarm) are better when lower (i.e. more negative delta).
    higher_is_better = (error_type == "correct")
    if avg_deltas:
        best_label = (max(avg_deltas, key=avg_deltas.get) if higher_is_better
                      else min(avg_deltas, key=avg_deltas.get))
    else:
        best_label = None

    if ylim is not None:
        ax.set_ylim(*ylim)
    _apply_dense_grid(ax, ylim=ylim, major_step=major_gridline_step, minor_step=minor_gridline_step)

    ax.set_ylabel(f"{error_type.title()} ({metric})")
    fig.suptitle(title or f"{error_type.title()} ({metric}) — all models vs {baseline_label}")

    if dax is not None:
        dax.axhline(0, color="black", linewidth=1, linestyle="--", zorder=2)
        dax.set_ylabel(f"Δ vs {baseline_label} ({unit})")
        dax.set_xlabel(x_label)
        dax.set_xticks(bar_centres)
        dax.set_xticklabels(tick_labels, rotation=45, ha="right")
        if delta_ylim is not None:
            dax.set_ylim(*delta_ylim)
        _apply_dense_grid(dax, ylim=delta_ylim, major_step=delta_major_gridline_step,
                            minor_step=delta_minor_gridline_step, default_major=2)

    all_handles = bar_handles[::-1] + model_lines
    all_labels = bar_labels[::-1] + [l.get_label() for l in model_lines]
    legend = ax.legend(
        all_handles, all_labels, title="Model", loc="upper left",
        bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8, ncol=legend_ncol,
    )
    if best_label is not None:
        for legend_text in legend.get_texts():
            if legend_text.get_text().startswith(f"{best_label} ("):
                legend_text.set_fontweight("bold")

    if dax is not None:
        # Explicit margins instead of tight_layout: with a forced rect bottom
        # of 0, tight_layout has no room left for the delta panel's rotated
        # x-tick labels and ends up compressing/clipping its y-axis labels.
        fig.subplots_adjust(left=0.06, right=0.78, top=0.92, bottom=0.18, hspace=0.12)
    else:
        plt.tight_layout(rect=[0, 0, 0.78, 1])  # reserve room on the right for the legend
    plt.show()