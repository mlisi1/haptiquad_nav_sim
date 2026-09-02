"""
timing_plotting.py — generates plots from a completed (or in-progress)
timing_sweep_orchestrator.py results directory
(<results_dir>/<tag>/csv/summary.csv).

Independent from plotting.py (the mass/mu classification sweep's plotting
module) -- different summary.csv schema (see timing_trial_logger.py's
SUMMARY_FIELDS), different question (task time vs. obstacle length, not
classification accuracy vs. mass/friction). No ROS import here either, so
it runs standalone against any past timing-sweep results.

Two plots, both answering the same question from timing_sweep_orchestrator.py's
docstring -- does tactile-probe navigation end up faster or slower than
classical circumvent-everything navigation, and how does that depend on the
obstacle's lateral length relative to the robot:

  time_vs_length.png    — mean total_time_s (± std across repeats) per arm
                          (classical/movable/solid) vs. length. The
                          "crossover" claim is whatever these three curves
                          show directly.
  timing_deltas.png     — benefit = classical - movable (should grow with
                          length: bigger detour avoided) and overhead =
                          solid - classical (should stay roughly FLAT: the
                          probe/backup/replan sequence is meant to be a
                          fixed tax, not one that grows with obstacle size).
                          A non-flat overhead curve is a red flag worth
                          investigating before trusting the length sweep.
  segment_breakdown.png — mean per-segment duration (pre-contact, probe-to-
                          classify, backup, post-classify-to-finish) vs.
                          length, movable and solid arms side by side --
                          the decomposition behind the total_time_s numbers
                          above.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

ARM_COLORS = {
    'classical': 'tab:gray',
    'movable': 'tab:green',
    'solid': 'tab:red',
}
ARM_ORDER = ('classical', 'movable', 'solid')

SEGMENT_FIELDS = [
    ('t_pre_contact_s', 'approach → contact'),
    ('t_probe_to_classify_exit_s', 'probe → classify'),
    ('t_backup_s', 'backup'),
    ('t_post_classify_to_finish_s', 'post-classify → goal'),
]


def _load_summary(tag_dir):
    path = os.path.join(tag_dir, 'csv', 'summary.csv')
    df = pd.read_csv(path)
    df = df[df['end_reason'] != 'superseded']  # forced-close artifacts, not real trials
    return df


def _grouped_stats(df, value_col):
    """Return {arm: (lengths_sorted, means, stds)} for a given column."""
    out = {}
    for arm in ARM_ORDER:
        sub = df[df['trial_mode'] == arm]
        if sub.empty:
            continue
        g = sub.groupby('size_y')[value_col].agg(['mean', 'std', 'count']).sort_index()
        out[arm] = (g.index.values, g['mean'].values, g['std'].fillna(0.0).values)
    return out


def plot_time_vs_length(df, out_path):
    stats = _grouped_stats(df, 'total_time_s')
    if not stats:
        raise ValueError('no rows to plot (empty summary.csv?)')

    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, (lengths, means, stds) in stats.items():
        ax.errorbar(lengths, means, yerr=stds, marker='o', capsize=3,
                    label=arm, color=ARM_COLORS.get(arm))
    ax.set_xlabel('Obstacle lateral length (m)')
    ax.set_ylabel('Total time to goal (s)')
    ax.set_title('Task completion time vs. obstacle length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timing_deltas(df, out_path):
    stats = _grouped_stats(df, 'total_time_s')
    if 'classical' not in stats:
        raise ValueError('no classical-arm rows -- deltas need a classical baseline')

    c_lengths, c_means, _ = stats['classical']
    c_by_len = dict(zip(c_lengths, c_means))

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    if 'movable' in stats:
        m_lengths, m_means, _ = stats['movable']
        common = [L for L in m_lengths if L in c_by_len]
        benefit = [c_by_len[L] - m for L, m in zip(m_lengths, m_means) if L in c_by_len]
        if common:
            ax.plot(common, benefit, marker='o', color=ARM_COLORS['movable'],
                    label='benefit = classical − movable')
            plotted = True
    if 'solid' in stats:
        s_lengths, s_means, _ = stats['solid']
        common = [L for L in s_lengths if L in c_by_len]
        overhead = [s - c_by_len[L] for L, s in zip(s_lengths, s_means) if L in c_by_len]
        if common:
            ax.plot(common, overhead, marker='o', color=ARM_COLORS['solid'],
                    label='overhead = solid − classical')
            plotted = True
    if not plotted:
        raise ValueError('no movable/solid rows sharing a length with the classical arm')

    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_xlabel('Obstacle lateral length (m)')
    ax.set_ylabel('Δ time vs. classical (s)')
    ax.set_title('Benefit (pushing) vs. overhead (probing a solid), relative to classical nav')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_segment_breakdown(df, out_path):
    arms = [a for a in ('movable', 'solid') if a in df['trial_mode'].unique()]
    if not arms:
        raise ValueError('no movable/solid rows to decompose (classical has no probe segments)')

    fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 5), squeeze=False)
    axes = axes[0]

    for ax, arm in zip(axes, arms):
        sub = df[df['trial_mode'] == arm]
        lengths = np.sort(sub['size_y'].unique())
        bottoms = np.zeros(len(lengths))
        for col, label in SEGMENT_FIELDS:
            if col not in sub.columns:
                continue
            means = sub.groupby('size_y')[col].mean().reindex(lengths).fillna(0.0).values
            ax.bar(lengths, means, bottom=bottoms, width=min(0.3, np.min(np.diff(lengths)) * 0.6
                                                             if len(lengths) > 1 else 0.3),
                  label=label)
            bottoms += means
        ax.set_xlabel('Obstacle lateral length (m)')
        ax.set_ylabel('Mean segment duration (s)')
        ax.set_title('%s arm' % arm)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


PLOTS = [
    ('time_vs_length.png', plot_time_vs_length),
    ('timing_deltas.png', plot_timing_deltas),
    ('segment_breakdown.png', plot_segment_breakdown),
]


def generate_all(tag_dir):
    """Writes every plot into <tag_dir>/plots/. Returns (written, errors)
    where written is a list of file paths and errors is a list of
    (filename, exception) tuples -- one bad/missing column skips only that
    plot, mirroring plotting.py's generate_all()."""
    df = _load_summary(tag_dir)
    plots_dir = os.path.join(tag_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    written, errors = [], []
    for filename, fn in PLOTS:
        out_path = os.path.join(plots_dir, filename)
        try:
            fn(df, out_path)
            written.append(out_path)
        except Exception as e:
            errors.append((filename, e))
    return written, errors
