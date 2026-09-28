"""
Visualization for CategoryStats and CategoryErrorAnalysis.

Every function here for an already computed CategoryErrorAnalysis or CategoryStats: it
reads .durations / .error_distribution / .confusion_durations (etc.) but
never calls compute_durations() or compute_distributions() itself.

Call these explicitly once, then customize plots as many times as you'd like.

---------------------------------------------------------------------------
Example usage
---------------------------------------------------------------------------

from category_metrics import CategoryStats, CategoryErrorAnalysis

--- plot_percentages / plot_durations: CategoryStats, no hypothesis needed ---

stats = CategoryStats(files=files, speaker_annotation="reference", category_annotation="gender")
stats.compute_durations(overlap=True)

plot_percentages(stats)   # pie chart of speech % per category label
plot_durations(stats)     # horizontal bar chart of speech duration per category label

--- plot_confusion: heatmap of which categories get confused with which ---

analysis = CategoryErrorAnalysis(files=files, speaker_annotation="reference",
                                  hypothesis_annotation="precision", category_annotation="gender")
analysis.compute_distributions(normalized=True)

plot_confusion(analysis)                       # all-speech confusion matrix
plot_confusion(analysis, for_overlap=True)      # confusion restricted to overlap regions

--- plot_analysis: full breakdown by error type, with optional color overrides ---

#show_confusion_split shows distribution of inter/intra confusion errors
#show_overlap shows % of errors occurring in regions of overlapping speech
plot_analysis(analysis, show_confusion_split=True, show_overlap=True,
              title="Error breakdown by gender",
              colors={"male": "#1f77b4", "female": "#d62728"})
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# CategoryStats plots
# ---------------------------------------------------------------------------

COLORS = [
    '#0000A2', '#BC272D', '#E9C716', '#50AD9F',
    '#7B4F9E', '#2E8B57', '#FF6B35', '#1B9AAA',
]


def _get_plot_colors(n):
    return [COLORS[i % len(COLORS)] for i in range(n)]


def _require_durations(stats):
    if stats.durations is None:
        raise RuntimeError("stats.compute_durations() must be called before plotting")


def plot_percentages(stats):
    """Pie chart of speech percentage per category label.

    stats: a CategoryStats instance with compute_durations() already called.
    """
    _require_durations(stats)
    percentages = stats.all_percentages()
    labels = stats.labels
    values = [percentages[l] for l in labels]
    colors = _get_plot_colors(len(labels))

    title = (
        "Percentage of total speech time per category (overlap included)"
        if stats.durations_overlap else
        "Percentage of timeline duration per category"
    )

    plt.close('all')
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
    ax.set_title(title)
    plt.show()


def plot_durations(stats):
    """Horizontal bar chart of speech duration per category label.

    stats: a CategoryStats instance with compute_durations() already called.
    """
    _require_durations(stats)
    raw = stats.all_durations()
    labels = stats.labels + ['Total']
    values = [raw[l] for l in stats.labels] + [raw['TOTAL']]
    colors = _get_plot_colors(len(labels))

    title = (
        "Duration of total speech time per category (overlap included)"
        if stats.durations_overlap else
        "Duration of speech in timeline per category"
    )

    fig, ax = plt.subplots()
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Duration of Speech Time (seconds)")
    ax.set_title(title)
    plt.show()


# ---------------------------------------------------------------------------
# CategoryErrorAnalysis plots
# ---------------------------------------------------------------------------

_AUTO_PALETTE = [
    '#008dff', '#ff9d3a', '#4ecb8d', '#e94f5b',
    '#9b5de5', '#f15bb5', '#fee440', '#00bbf9',
    '#0000A2', '#BC272D', '#E9C716', '#50AD9F',
]


def _build_color_map(labels, user_colors=None):
    """Returns {label: hex_color} for every label, with optional per-label overrides."""
    color_map = {}
    palette_idx = 0
    for label in labels:
        if user_colors and label in user_colors:
            color_map[label] = user_colors[label]
        else:
            color_map[label] = _AUTO_PALETTE[palette_idx % len(_AUTO_PALETTE)]
            palette_idx += 1
    return color_map


def _require_distributions(analysis):
    if analysis.error_distribution is None:
        raise RuntimeError("analysis.compute_distributions() must be called before plotting")



def plot_confusion(analysis, for_overlap=False, ax=None):
    """Heatmap of normalized confusion durations.

    analysis: a CategoryErrorAnalysis instance with compute_distributions()
              already called.
    for_overlap: if True, plots overlap_confusion_durations instead of
                 confusion_durations.
    ax: external Axes for embedding in multi-panel figures.
    """
    _require_distributions(analysis)
    durations = (analysis.overlap_confusion_durations
                 if for_overlap else analysis.confusion_durations)

    matrix = np.array([
        [durations.get(r + '+' + p, 0) for p in analysis.labels]
        for r in analysis.labels
    ])

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots()

    sns.heatmap(matrix, annot=True, fmt='.2f',
                xticklabels=analysis.labels, yticklabels=analysis.labels,
                cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Reference')

    if own_fig:
        plt.tight_layout()
        plt.show()


def plot_analysis(analysis, ax=None,
                  title=None, show_legend=True, show_overlap=True,
                  show_confusion_split=True, show_unknown=False, colors=None):
    """
    Bar chart of the proportion of each category's speech affected by each error type.

    analysis: a CategoryErrorAnalysis instance with compute_distributions()
              already called.

    Note: labels that have zero values across all error types (md, fa, confusion)
    are dropped from the plot. If a label is absent from the chart,
    it had no recorded errors in the evaluated data.

    ax                   : external Axes for embedding in multi-panel figures
    title                : custom plot title
    show_legend          : if False, suppresses the legend
    show_overlap         : if True, hatches the overlap-region portion of each bar
    show_confusion_split : if True, splits confusion bars into same- vs cross-category
    show_unknown         : if True, adds a grey segment for unmapped speakers
    colors               : dict {label: hex str} or None -- per-label color overrides
    """
    
    # ── Data — already computed, just read it ──────────────────────────────
    _require_distributions(analysis)
    error_dist       = analysis.error_distribution
    conf_dur         = analysis.confusion_durations
    overlap_dist     = analysis.overlap_error_distribution
    overlap_conf_dur = analysis.overlap_confusion_durations
    color_map        = _build_color_map(analysis.labels, user_colors=colors)

    # Only keep labels that have non-zero values
    plot_labels = [
        l for l in analysis.labels
        if any(error_dist[err].get(l, 0) > 0 for err in ('md', 'fa'))
        or error_dist['confusion']['ref'].get(l, 0) > 0
    ]

    # ── Constants ─────────────────────────────────────────────────────────
    ERR_TYPES     = ['md', 'fa', 'confusion']
    UNKNOWN_COLOR = '#999999'
    LIGHT_FACTOR  = 0.35
    DARK_FACTOR   = 0.35
    HATCH         = '//'

    # ── Color helpers ─────────────────────────────────────────────────────

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

    def same_frac(label, dist):
        return dist.get(label + '+' + label, 0.0)

    def cross_frac(label, dist):
        return sum(
            v for k, v in dist.items()
            if k.startswith(label + '+')
            and not k.endswith('+UNKNOWN')
            and k != label + '+' + label
        )

    def unknown_frac(label, dist):
        return dist.get(label + '+UNKNOWN', 0.0)

    # ── Label helpers ──────────────────────────────────────────────────────

    def side_sub_label(ax, x, y_bottom, height, pct, bar_w, dx=0.015, color='#333333'):
        if height < 1e-4:
            return
        ax.text(x + bar_w / 2 + dx, y_bottom + height / 2,
                f'{pct * 100:.1f}%',
                ha='left', va='center', fontsize=8.5, color=color, zorder=6)

    def sub_label(ax, x, y_bottom, height, pct, bar_w, color='#333333'):
        if height < 0.003:
            return
        ax.text(x, y_bottom + height + 0.001, f'{pct * 100:.1f}%',
                ha='center', va='bottom', fontsize=7, color='white', zorder=6)

    # ── Layout ────────────────────────────────────────────────────────────

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(14, 6))

    n_labels      = len(plot_labels)
    group_w       = 1.2
    intra_gap     = 0.5
    bar_w         = group_w / (n_labels + (n_labels - 1) * intra_gap)
    group_spacing = 1.8
    group_centers = np.arange(len(ERR_TYPES)) * group_spacing

    # ── Draw bars ─────────────────────────────────────────────────────────

    for li, label in enumerate(plot_labels):
        color     = color_map.get(label, _AUTO_PALETTE[li % len(_AUTO_PALETTE)])
        light_col = lighten(color)
        dark_col  = darken(color)
        step      = bar_w * (1 + intra_gap)
        offset    = (li - (n_labels - 1) / 2) * step
        xs        = group_centers + offset

        for ei, err in enumerate(ERR_TYPES):
            x = xs[ei]

            # ── confusion ─────────────────────────────────────────────────
            if err == 'confusion':
                total_conf = error_dist['confusion']['ref'].get(label, 0)

                if show_confusion_split:
                    same_r  = same_frac(label, conf_dur)
                    cross_r = cross_frac(label, conf_dur)
                    unk_r   = unknown_frac(label, conf_dur)
                    denom   = same_r + cross_r + unk_r
                    scale   = total_conf / denom if denom > 0 else 0

                    same_h  = same_r  * scale
                    cross_h = cross_r * scale
                    unk_h   = unk_r   * scale if show_unknown else 0.0

                    ax.bar(x, same_h, width=bar_w * 0.9,
                           color=light_col, edgecolor='white', linewidth=0.4, zorder=3)
                    side_sub_label(ax, x, 0, same_h, same_h, bar_w, color=light_col)

                    ax.bar(x, cross_h, width=bar_w * 0.9, bottom=same_h,
                           color=dark_col, edgecolor='white', linewidth=0.4, zorder=3)
                    side_sub_label(ax, x, same_h, cross_h, cross_h, bar_w, color=dark_col)

                    if show_unknown and unk_h > 0:
                        ax.bar(x, unk_h, width=bar_w * 0.9, bottom=same_h + cross_h,
                               color=UNKNOWN_COLOR, edgecolor='white', linewidth=0.4, zorder=3)
                        side_sub_label(ax, x, same_h + cross_h, unk_h, unk_h,
                                       bar_w, color=UNKNOWN_COLOR)

                    if show_overlap:
                        ol_total  = overlap_dist['confusion']['ref'].get(label, 0)
                        ol_same_r = same_frac(label, overlap_conf_dur)
                        ol_cross_r= cross_frac(label, overlap_conf_dur)
                        ol_unk_r  = unknown_frac(label, overlap_conf_dur)
                        ol_denom  = ol_same_r + ol_cross_r + ol_unk_r
                        ol_scale  = ol_total / ol_denom if ol_denom > 0 else 0

                        ol_same_h  = ol_same_r  * ol_scale
                        ol_cross_h = ol_cross_r * ol_scale
                        ol_unk_h   = ol_unk_r   * ol_scale if show_unknown else 0.0

                        if ol_same_h > 0:
                            ax.bar(x, ol_same_h, width=bar_w * 0.9, bottom=0,
                                   color=light_col, edgecolor='white',
                                   hatch=HATCH, linewidth=0.5, zorder=4)
                            sub_label(ax, x, 0, ol_same_h, ol_same_h, bar_w, color=color)

                        if ol_cross_h > 0:
                            ax.bar(x, ol_cross_h, width=bar_w * 0.9, bottom=same_h,
                                   color=dark_col, edgecolor='white',
                                   hatch=HATCH, linewidth=0.5, zorder=4)
                            sub_label(ax, x, same_h, ol_cross_h, ol_cross_h, bar_w, color=color)

                        if show_unknown and ol_unk_h > 0:
                            ax.bar(x, ol_unk_h, width=bar_w * 0.9,
                                   bottom=same_h + cross_h,
                                   color=UNKNOWN_COLOR, edgecolor='white',
                                   hatch=HATCH, linewidth=0.5, zorder=4)
                            sub_label(ax, x, same_h + cross_h, ol_unk_h, ol_unk_h,
                                      bar_w, color=UNKNOWN_COLOR)

                    bar_top = same_h + cross_h + (unk_h if show_unknown else 0)

                else:
                    if show_unknown:
                        bar_val = total_conf
                    else:
                        unk_r   = unknown_frac(label, conf_dur)
                        denom   = same_frac(label, conf_dur) + cross_frac(label, conf_dur) + unk_r
                        unk_h   = unk_r / denom * total_conf if denom > 0 else 0
                        bar_val = total_conf - unk_h

                    ax.bar(x, bar_val, width=bar_w * 0.9,
                           color=color, edgecolor='white', linewidth=0.4, zorder=3)

                    if show_overlap:
                        ol_val = overlap_dist['confusion']['ref'].get(label, 0)
                        if ol_val > 0:
                            ax.bar(x, ol_val, width=bar_w * 0.9, bottom=0,
                                   color=color, edgecolor='white',
                                   hatch=HATCH, linewidth=0.5, zorder=4)

                    bar_top = bar_val

            # ── md / fa ───────────────────────────────────────────────────
            else:
                val = error_dist[err].get(label, 0)
                ax.bar(x, val, width=bar_w * 0.9,
                       color=color, edgecolor='white', linewidth=0.4, zorder=3)

                if show_overlap and err in ('md', 'fa'):
                    ol_val = overlap_dist[err].get(label, 0)
                    if ol_val > 0:
                        ax.bar(x, ol_val, width=bar_w * 0.9, bottom=0,
                               color=color, edgecolor='white',
                               hatch=HATCH, linewidth=0.5, zorder=4)
                        sub_label(ax, x, 0, ol_val, ol_val, bar_w, color=color)

                bar_top = val

            ax.text(x, bar_top + 0.001, f'{bar_top * 100:.1f}%',
                    ha='center', va='bottom', fontsize=9,
                    color='#333333', fontweight='bold', zorder=5)

    # ── Legend ────────────────────────────────────────────────────────────

    legend_handles = [
        mpatches.Patch(color=color_map.get(label, '#888888'), label=label)
        for label in plot_labels
    ]

    if show_confusion_split:
        legend_handles += [
            mpatches.Patch(color='#cccccc', label='Same-category confusion'),
            mpatches.Patch(color='#333333', label='Cross-category confusion'),
        ]
        if show_unknown:
            legend_handles.append(
                mpatches.Patch(color=UNKNOWN_COLOR, label='Unknown speaker confusion'))

    if show_overlap:
        legend_handles.append(
            mpatches.Patch(facecolor='none', edgecolor='#444444',
                           hatch=HATCH, label='Occurring in overlap regions'))

    if show_legend:
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
    ax.set_title(
        title or 'Proportion of Category Speech Affected by Error Type',
        fontsize=13, fontweight='bold', pad=12,
    )
    padding = 0.6
    ax.set_xlim(group_centers[0] - padding, group_centers[-1] + padding)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if own_fig:
        plt.tight_layout()
        plt.show()