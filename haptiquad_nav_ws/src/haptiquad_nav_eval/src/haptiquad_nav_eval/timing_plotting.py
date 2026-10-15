"""
timing_plotting.py — generates plots from a completed (or in-progress)
timing_sweep_orchestrator.py results directory
(<results_dir>/<tag>/csv/summary.csv).

Independent from plotting.py (the mass/mu classification sweep's plotting
module) -- different summary.csv schema (see timing_trial_logger.py's
SUMMARY_FIELDS), different question (task time vs. obstacle length, not
classification accuracy vs. mass/friction). No ROS import here either, so
it runs standalone against any past timing-sweep results.

Config-driven, one YAML file per plot, same mechanism as plotting.py (see
that module's docstring): generate_all() writes (or, on a re-run, reads
back and reuses) <tag_dir>/plots/config/<plot_name>.yaml for every plot via
the shared plot_config.load_or_init_config() -- edit any of those files and
re-run to change just that plot's title/axis labels/legend text/colors/
figure size/etc, no simulation, no ROS, no rebuild required. See
default_config() below for every tunable and its default. generate_all()
also bundles this module + plot_config.py + a tiny standalone runner into
<tag_dir>/plot_scripts/, so the results folder is self-contained -- see
plotting.py's module docstring for the full rationale, identical here.

Six plots, all answering the same question from
timing_sweep_orchestrator.py's docstring -- does tactile-probe navigation
end up faster or slower than classical circumvent-everything navigation,
and how does that depend on the obstacle's lateral length relative to the
robot. Each of the three questions below has a "_no_solid" companion plot
that drops the solid arm entirely (config key 'arms' / 'include_solid') --
solid's much larger, flat time cost dwarfs the classical/movable crossover
on a shared axis, so a plot focused on just "does pushing beat
circumventing" reads better without it:

  time_vs_length[_no_solid].png    — mean total_time_s (± std across
                          repeats) per arm (classical/movable/solid) vs.
                          length. The "crossover" claim is whatever these
                          curves show directly.
  timing_deltas[_no_solid].png     — benefit = classical - movable (should
                          grow with length: bigger detour avoided) and
                          overhead = solid - classical (should stay
                          roughly FLAT: the probe/backup/replan sequence is
                          meant to be a fixed tax, not one that grows with
                          obstacle size). A non-flat overhead curve is a
                          red flag worth investigating before trusting the
                          length sweep.
  segment_breakdown[_no_solid].png — mean per-segment duration (approach →
                          contact, probe → classify, backup, post-classify
                          → goal) vs. length, movable and solid arms side
                          by side -- the decomposition behind the
                          total_time_s numbers above.

Besides the sim_anomaly filter (SimHealthMonitor's online stopped/tip-over/
teleport detection), load_summary() also applies a statistical duration-
outlier filter (config/outlier_filter.yaml, see
plot_config.exclude_duration_outliers()'s docstring) -- confirmed necessary
by a live run (test_timing_opt_speed) where several movable trials took
4-10x their arm's median total_time_s with no sim_anomaly ever flagged (the
robot kept moving, just far slower/more erratically than normal, which the
online zero-movement stall check cannot see). Increasing `repeats` in
timing_sweep.yaml makes each arm's per-length mean less sensitive to any
one remaining unfiltered outlier too.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

try:
    # Normal case: imported as haptiquad_nav_eval.timing_plotting.
    from haptiquad_nav_eval.plot_config import (load_or_init_config, bundle_scripts,
                                                exclude_duration_outliers)
except ImportError:
    # Self-contained plot_scripts/ bundle case (see bundle_scripts() below):
    # plot_config.py sits flat next to this file with no package wrapper,
    # and regenerate.py has already put that directory on sys.path.
    from plot_config import load_or_init_config, bundle_scripts, exclude_duration_outliers

ARM_COLORS = {
    'classical': 'tab:gray',
    'movable': 'tab:green',
    'solid': 'tab:red',
}
ARM_ORDER = ('classical', 'movable', 'solid')

# (column, default legend/segment label). Column set and stacking ORDER are
# fixed here (they're the physical sequence of a trial); default_config()
# exposes the LABEL TEXT and which of these columns to include/how to order
# them per plot via each section's 'segments' + 'segment_labels' keys.
SEGMENT_FIELDS = [
    ('t_pre_contact_s', 'approach → contact'),
    ('t_probe_to_classify_exit_s', 'probe → classify'),
    ('t_backup_s', 'backup'),
    ('t_post_classify_to_finish_s', 'post-classify → goal'),
]

LENGTH_AXIS_LABEL = 'Obstacle lateral length (× robot width)'


# ── Default configuration ────────────────────────────────────────────────────
# One key per plot (matching its output filename stem). Nothing here is read
# directly by the plot functions at call time except via the `cfg` argument
# they're passed -- see plot_config.load_or_init_config()'s docstring.

def default_config():
    return {
        'outlier_filter': {
            'enabled': True,
            # total_time_s is the metric that actually blew up in the
            # incident this filter exists for (see module docstring) --
            # per-segment columns can still look individually fine on a
            # trial whose total nonetheless ends up way outside its arm's
            # normal range.
            'value_col': 'total_time_s',
            'group_by': ['trial_mode'],   # median computed per arm
            'multiplier': 3.0,            # flag total_time_s > multiplier * group median
            'min_group_size': 3,          # groups smaller than this are never filtered
        },
        'time_vs_length': {
            'arms': list(ARM_ORDER),
            'figsize': [7, 5],
            'dpi': 150,
            'title': 'Task completion time vs. obstacle length',
            'xlabel': LENGTH_AXIS_LABEL,
            'ylabel': 'Total time to goal (s)',
            'legend_fontsize': None,   # null -> matplotlib default
            # Per-arm color override, merged over the built-in ARM_COLORS
            # (classical/movable/solid).
            'colors': {},
        },
        'time_vs_length_no_solid': {
            'arms': ['classical', 'movable'],
            'figsize': [7, 5],
            'dpi': 150,
            'title': 'Task completion time vs. obstacle length',
            'xlabel': LENGTH_AXIS_LABEL,
            'ylabel': 'Total time to goal (s)',
            'legend_fontsize': None,
            'colors': {},
        },
        'timing_deltas': {
            'include_solid': True,
            'figsize': [7, 5],
            'dpi': 150,
            'title': 'Benefit (pushing) vs. overhead (probing a solid), relative to classical nav',
            'xlabel': LENGTH_AXIS_LABEL,
            'ylabel': 'Δ time vs. classical (s)',
            'benefit_label': 'benefit = classical − movable',
            'overhead_label': 'overhead = solid − classical',
            'legend_fontsize': None,
            'colors': {},   # e.g. {movable: ..., solid: ...} overrides the curve colors
        },
        'timing_deltas_no_solid': {
            'include_solid': False,
            'figsize': [7, 5],
            'dpi': 150,
            'title': 'Benefit of pushing through a movable obstacle, relative to classical nav',
            'xlabel': LENGTH_AXIS_LABEL,
            'ylabel': 'Δ time vs. classical (s)',
            'benefit_label': 'benefit = classical − movable',
            'overhead_label': 'overhead = solid − classical',
            'legend_fontsize': None,
            'colors': {},
        },
        'segment_breakdown': {
            'arms': ['movable', 'solid'],
            'segments': [c for c, _ in SEGMENT_FIELDS],   # which columns, and in what stacking order
            'segment_labels': dict(SEGMENT_FIELDS),        # per-column legend text override
            'figsize_per_arm': [6, 5],
            'dpi': 150,
            'xlabel': LENGTH_AXIS_LABEL,
            'ylabel': 'Mean segment duration (s)',
            'legend_fontsize': 8,
            'bar_width_frac': 0.6,
        },
        'segment_breakdown_no_solid': {
            'arms': ['movable'],
            'segments': [c for c, _ in SEGMENT_FIELDS],
            'segment_labels': dict(SEGMENT_FIELDS),
            'figsize_per_arm': [6, 5],
            'dpi': 150,
            'xlabel': LENGTH_AXIS_LABEL,
            'ylabel': 'Mean segment duration (s)',
            'legend_fontsize': 8,
            'bar_width_frac': 0.6,
        },
    }


# ── Data loading ─────────────────────────────────────────────────────────────

def load_summary(tag_dir, config=None):
    path = os.path.join(tag_dir, 'csv', 'summary.csv')
    df = pd.read_csv(path)
    n_before = len(df)
    df = df[df['end_reason'] != 'superseded']  # forced-close artifacts, not real trials
    # sim_anomaly (SimHealthMonitor, trial_core.py): a Gazebo physics
    # explosion/teleport, tip-over, out-of-range height, or a stall was
    # detected during the trial. Older summary.csv files predate this
    # column; skip the filter rather than KeyError on them.
    if 'sim_anomaly' in df.columns:
        df = df[df['sim_anomaly'].fillna('') == '']
    n_dropped = n_before - len(df)
    if n_dropped:
        print('[timing_plotting] Excluded %d/%d rows (superseded/sim_anomaly) from %s' %
              (n_dropped, n_before, tag_dir))

    ocfg = (config or {}).get('outlier_filter', {})
    if ocfg.get('enabled', True):
        n_before_outliers = len(df)
        df, dropped = exclude_duration_outliers(
            df, ocfg.get('group_by', ['trial_mode']), ocfg.get('value_col', 'total_time_s'),
            multiplier=ocfg.get('multiplier', 3.0), min_group_size=ocfg.get('min_group_size', 3))
        if len(dropped):
            print('[timing_plotting] Excluded %d/%d rows as duration outliers from %s:' %
                  (len(dropped), n_before_outliers, tag_dir))
            for _, row in dropped.iterrows():
                print('    trial_id=%s trial_mode=%s %s=%.2f' %
                     (row['trial_id'], row['trial_mode'], ocfg.get('value_col', 'total_time_s'),
                      row[ocfg.get('value_col', 'total_time_s')]))

    # size_y is logged in absolute meters (needed to spawn the object); express
    # it in robot-width units for plotting since that's the physically
    # meaningful scale for "can this be circumvented" (see timing_sweep.yaml's
    # robot_width_m comment). robot_width_m is logged per-row rather than
    # hardcoded here so older summary.csv files without the column fail
    # loudly instead of silently plotting raw meters as if they were widths.
    df = df.copy()
    df['length_rw'] = df['size_y'] / df['robot_width_m']
    return df


def _grouped_stats(df, value_col, arms):
    """Return {arm: (lengths_sorted, means, stds)} for a given column."""
    out = {}
    for arm in arms:
        sub = df[df['trial_mode'] == arm]
        if sub.empty:
            continue
        g = sub.groupby('length_rw')[value_col].agg(['mean', 'std', 'count']).sort_index()
        out[arm] = (g.index.values, g['mean'].values, g['std'].fillna(0.0).values)
    return out


def plot_time_vs_length(df, out_path, cfg):
    arms = cfg.get('arms', list(ARM_ORDER))
    colors = dict(ARM_COLORS, **cfg.get('colors', {}))
    stats = _grouped_stats(df, 'total_time_s', arms)
    if not stats:
        raise ValueError('no rows to plot -- empty summary.csv, or none of the arms configured in '
                         'config/%s.yaml are present in this results directory'
                         % os.path.splitext(os.path.basename(out_path))[0])

    fig, ax = plt.subplots(figsize=cfg.get('figsize', (7, 5)))
    for arm, (lengths, means, stds) in stats.items():
        ax.errorbar(lengths, means, yerr=stds, marker='o', capsize=3,
                    label=arm, color=colors.get(arm))
    ax.set_xlabel(cfg.get('xlabel', LENGTH_AXIS_LABEL))
    ax.set_ylabel(cfg.get('ylabel', 'Total time to goal (s)'))
    ax.set_title(cfg.get('title', 'Task completion time vs. obstacle length'))
    ax.legend(fontsize=cfg.get('legend_fontsize'))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
    plt.close(fig)


def plot_timing_deltas(df, out_path, cfg):
    include_solid = cfg.get('include_solid', True)
    colors = dict(ARM_COLORS, **cfg.get('colors', {}))
    stats = _grouped_stats(df, 'total_time_s', ARM_ORDER)
    if 'classical' not in stats:
        raise ValueError('no classical-arm rows -- deltas need a classical baseline')

    c_lengths, c_means, _ = stats['classical']
    c_by_len = dict(zip(c_lengths, c_means))

    fig, ax = plt.subplots(figsize=cfg.get('figsize', (7, 5)))
    plotted = False
    if 'movable' in stats:
        m_lengths, m_means, _ = stats['movable']
        common = [L for L in m_lengths if L in c_by_len]
        benefit = [c_by_len[L] - m for L, m in zip(m_lengths, m_means) if L in c_by_len]
        if common:
            ax.plot(common, benefit, marker='o', color=colors['movable'],
                    label=cfg.get('benefit_label', 'benefit = classical − movable'))
            plotted = True
    if include_solid and 'solid' in stats:
        s_lengths, s_means, _ = stats['solid']
        common = [L for L in s_lengths if L in c_by_len]
        overhead = [s - c_by_len[L] for L, s in zip(s_lengths, s_means) if L in c_by_len]
        if common:
            ax.plot(common, overhead, marker='o', color=colors['solid'],
                    label=cfg.get('overhead_label', 'overhead = solid − classical'))
            plotted = True
    if not plotted:
        raise ValueError('no movable/solid rows sharing a length with the classical arm')

    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_xlabel(cfg.get('xlabel', LENGTH_AXIS_LABEL))
    ax.set_ylabel(cfg.get('ylabel', 'Δ time vs. classical (s)'))
    default_title = ('Benefit (pushing) vs. overhead (probing a solid), relative to classical nav'
                     if include_solid else
                     'Benefit of pushing through a movable obstacle, relative to classical nav')
    ax.set_title(cfg.get('title', default_title))
    ax.legend(fontsize=cfg.get('legend_fontsize'))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
    plt.close(fig)


def plot_segment_breakdown(df, out_path, cfg):
    arms = [a for a in cfg.get('arms', ['movable', 'solid']) if a in df['trial_mode'].unique()]
    if not arms:
        raise ValueError('no rows for the arms configured in config/%s.yaml (classical has no '
                         'probe segments)' % os.path.splitext(os.path.basename(out_path))[0])
    segment_cols = cfg.get('segments', [c for c, _ in SEGMENT_FIELDS])
    labels = dict(dict(SEGMENT_FIELDS), **cfg.get('segment_labels', {}))
    width_frac = cfg.get('bar_width_frac', 0.6)

    per_w, h = cfg.get('figsize_per_arm', (6, 5))
    fig, axes = plt.subplots(1, len(arms), figsize=(per_w * len(arms), h), squeeze=False)
    axes = axes[0]

    for ax, arm in zip(axes, arms):
        sub = df[df['trial_mode'] == arm]
        lengths = np.sort(sub['length_rw'].unique())
        bottoms = np.zeros(len(lengths))
        width = (min(0.3, np.min(np.diff(lengths)) * width_frac) if len(lengths) > 1 else 0.3)
        for col in segment_cols:
            if col not in sub.columns:
                continue
            means = sub.groupby('length_rw')[col].mean().reindex(lengths).fillna(0.0).values
            ax.bar(lengths, means, bottom=bottoms, width=width, label=labels.get(col, col))
            bottoms += means
        ax.set_xlabel(cfg.get('xlabel', LENGTH_AXIS_LABEL))
        ax.set_ylabel(cfg.get('ylabel', 'Mean segment duration (s)'))
        ax.set_title('%s arm' % arm)
        ax.legend(fontsize=cfg.get('legend_fontsize', 8))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────

_THIS_FILE = os.path.abspath(__file__)
_PLOT_CONFIG_FILE = os.path.join(os.path.dirname(_THIS_FILE), 'plot_config.py')

_REGENERATE_SCRIPT = '''#!/usr/bin/env python3
"""Standalone regenerator for this results folder's plots. No ROS, no
rospkg needed. Just:

    python3 regenerate.py

Requires: pandas, numpy, matplotlib, pyyaml (plain pip packages).
Edit the YAML files in ../plots/config/ first to change any plot's
title/axis labels/legend text/colors/figure size, then re-run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_plotting import generate_all

if __name__ == '__main__':
    tag_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    written, errors = generate_all(tag_dir)
    for path in written:
        print('wrote', path)
    for filename, e in errors:
        print('SKIPPED %s: %s' % (filename, e), file=sys.stderr)
'''

# (filename, plot function, config section name)
PLOTS = [
    ('time_vs_length.png', plot_time_vs_length, 'time_vs_length'),
    ('time_vs_length_no_solid.png', plot_time_vs_length, 'time_vs_length_no_solid'),
    ('timing_deltas.png', plot_timing_deltas, 'timing_deltas'),
    ('timing_deltas_no_solid.png', plot_timing_deltas, 'timing_deltas_no_solid'),
    ('segment_breakdown.png', plot_segment_breakdown, 'segment_breakdown'),
    ('segment_breakdown_no_solid.png', plot_segment_breakdown, 'segment_breakdown_no_solid'),
]


def generate_all(tag_dir):
    """Writes every plot into <tag_dir>/plots/, per-plot config into
    <tag_dir>/plots/config/, and a self-contained script bundle into
    <tag_dir>/plot_scripts/ (see module docstring). Returns (written,
    errors) where written is a list of file paths and errors is a list of
    (filename, exception) tuples -- one bad/missing column skips only that
    plot, mirroring plotting.py's generate_all()."""
    config = load_or_init_config(tag_dir, default_config())
    df = load_summary(tag_dir, config)
    plots_dir = os.path.join(tag_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    written, errors = [], []
    for filename, fn, section in PLOTS:
        out_path = os.path.join(plots_dir, filename)
        try:
            fn(df, out_path, config[section])
            written.append(out_path)
        except Exception as e:
            errors.append((filename, e))

    bundle_scripts(tag_dir, [_THIS_FILE, _PLOT_CONFIG_FILE], _REGENERATE_SCRIPT)

    return written, errors
