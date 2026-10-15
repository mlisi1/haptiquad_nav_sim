"""
tuning_plotting.py — generates plots from a completed (or in-progress)
tuning_sweep_orchestrator.py results directory
(<results_dir>/<tag>/csv/summary.csv).

Independent from plotting.py (mass/mu sweep) and timing_plotting.py
(lateral-length sweep) -- different summary.csv schema, different question
(how contact-probe TIMING parameters affect classification time and
correctness, not classification accuracy vs. mass/friction or task time vs.
obstacle length). No ROS import here either, so it runs standalone against
any past tuning-sweep results.

Config-driven, one YAML file per plot, same mechanism as plotting.py (see
that module's docstring): generate_all() writes (or, on a re-run, reads
back and reuses) <tag_dir>/plots/config/<plot_name>.yaml for every plot via
the shared plot_config.load_or_init_config() -- edit any of those files and
re-run to change just that plot's titles/axis labels/legend text/colors/
figure size, no simulation, no ROS, no rebuild required. See
default_config() below for every tunable and its default. generate_all()
also bundles this module + plot_config.py + a tiny standalone runner into
<tag_dir>/plot_scripts/, so the results folder is self-contained -- see
plotting.py's module docstring for the full rationale, identical here.

Design note -- "vs param" plots use the WHOLE summary.csv, not just the rows
where a given parameter was the one being swept: tuning_sweep_orchestrator.py
runs a one-parameter-at-a-time (OFAT) design, so on every trial where
`param` was NOT being swept, it was still published at its baseline value
(never left unset) -- those rows are valid, if repeated, samples at that x.
Grouping directly on the column across every row therefore pools "the
entirety of the sweep" into each curve, exactly as intended, rather than
needlessly discarding most of the data for each individual plot.

For each of the six tunable contact-probe parameters, two files:
  time_vs_<param>.png         — the metric this sweep chiefly exists to
                                 characterize: time to DECISION
                                 (classify_time window itself), time
                                 PROBE-entry to DECISION (the actual
                                 "classification time", including
                                 approach-to-contact + contact_hold_time),
                                 and classification CORRECTNESS (fraction
                                 matching the expected label for that arm) --
                                 movable/solid overlaid, vs. that parameter's
                                 value.
  diagnostics_vs_<param>.png  — peak roll/pitch (stability) and mean wrench
                                 latency (directly relevant to
                                 wrench_filter_window, but shown for every
                                 parameter in case of cross-effects) vs. that
                                 parameter's value.
Plus one whole-sweep sanity check:
  diagnostics_over_sweep.png  — loop rate / wrench latency vs. trial_id, to
                                 confirm contact_probe_manager stayed healthy
                                 across the whole run.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

try:
    # Normal case: imported as haptiquad_nav_eval.tuning_plotting.
    from haptiquad_nav_eval.plot_config import (load_or_init_config, bundle_scripts,
                                                exclude_duration_outliers)
except ImportError:
    # Self-contained plot_scripts/ bundle case (see bundle_scripts() below):
    # plot_config.py sits flat next to this file with no package wrapper,
    # and regenerate.py has already put that directory on sys.path.
    from plot_config import load_or_init_config, bundle_scripts, exclude_duration_outliers

ARM_COLORS = {'movable': 'tab:green', 'solid': 'tab:red'}
ARM_ORDER = ('movable', 'solid')

SWEEP_PARAMS = [
    'probe_speed', 'probe_distance', 'contact_hold_time',
    'classify_time', 'classification_force_settle_time', 'wrench_filter_window',
]

PARAM_LABELS = {
    'probe_speed': 'probe_speed (m/s)',
    'probe_distance': 'probe_distance (m)',
    'contact_hold_time': 'contact_hold_time (s)',
    'classify_time': 'classify_time (s)',
    'classification_force_settle_time': 'classification_force_settle_time (s)',
    'wrench_filter_window': 'wrench_filter_window (samples)',
}


# ── Default configuration ────────────────────────────────────────────────────
# One key per plot (matching its output filename stem): time_vs_<param> and
# diagnostics_vs_<param> for each of the six SWEEP_PARAMS, plus
# diagnostics_over_sweep. Nothing here is read directly by the plot
# functions at call time except via the `cfg` argument they're passed -- see
# plot_config.load_or_init_config()'s docstring.

def default_config():
    cfg = {
        'outlier_filter': {
            'enabled': True,
            # t_probe_to_classify_exit_s is this sweep's own core metric
            # (the "classification time" it exists to characterize) and the
            # one that would blow up if a trial hit an undetected sim
            # glitch -- see plot_config.exclude_duration_outliers()'s
            # docstring for why this filter exists alongside sim_anomaly.
            'value_col': 't_probe_to_classify_exit_s',
            'group_by': ['trial_mode'],   # median computed per arm
            'multiplier': 3.0,
            'min_group_size': 3,
        },
    }
    for param in SWEEP_PARAMS:
        xlabel = PARAM_LABELS.get(param, param)
        cfg['time_vs_%s' % param] = {
            'figsize': [8, 11],
            'dpi': 150,
            'xlabel': xlabel,
            'decision_ylabel': 'time (s)',
            'decision_title': 'Time to classification DECISION (classify_time window itself)',
            'full_ylabel': 'time (s)',
            'full_title': 'Time PROBE-entry to DECISION\n'
                         '(the "classification time" this sweep tunes)',
            'full_title_fontsize': 10,
            'accuracy_ylabel': 'fraction correctly classified',
            'accuracy_title': 'Classification correctness vs. expected label',
            'legend_fontsize': 8,
            # Per-arm color override, merged over the built-in ARM_COLORS
            # (movable/solid).
            'colors': {},
        }
        cfg['diagnostics_vs_%s' % param] = {
            'figsize': [7, 8],
            'dpi': 150,
            'xlabel': xlabel,
            'stability_ylabel': 'peak |roll or pitch| (deg)',
            'stability_title': 'Base stability during the trial',
            'latency_ylabel': 'mean wrench latency (ms)',
            'latency_title': 'contact_probe_manager wrench latency',
            'legend_fontsize': 8,
            'colors': {},
        }
    cfg['diagnostics_over_sweep'] = {
        'figsize': [8, 6],
        'dpi': 150,
        'target_loop_rate_hz': 10.0,
        'loop_rate_color': 'tab:blue',
        'latency_color': 'tab:orange',
        'legend_fontsize': 8,
        'title': 'contact_probe_manager runtime diagnostics over the tuning sweep',
    }
    return cfg


def _load_summary(tag_dir, config=None):
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
        print('[tuning_plotting] Excluded %d/%d rows (superseded/sim_anomaly) from %s' %
              (n_dropped, n_before, tag_dir))

    ocfg = (config or {}).get('outlier_filter', {})
    if ocfg.get('enabled', True):
        n_before_outliers = len(df)
        df, dropped = exclude_duration_outliers(
            df, ocfg.get('group_by', ['trial_mode']),
            ocfg.get('value_col', 't_probe_to_classify_exit_s'),
            multiplier=ocfg.get('multiplier', 3.0), min_group_size=ocfg.get('min_group_size', 3))
        if len(dropped):
            value_col = ocfg.get('value_col', 't_probe_to_classify_exit_s')
            print('[tuning_plotting] Excluded %d/%d rows as duration outliers from %s:' %
                  (len(dropped), n_before_outliers, tag_dir))
            for _, row in dropped.iterrows():
                print('    trial_id=%s trial_mode=%s %s=%.2f' %
                     (row['trial_id'], row['trial_mode'], value_col, row[value_col]))
    return df


def _expected_label(trial_mode):
    return 'MOVABLE' if trial_mode == 'movable' else 'SOLID'


def plot_time_vs_param(df, param, out_path, cfg):
    """Three stacked panels for one tunable parameter -- see module
    docstring. Raises if `param` has no non-NaN values (e.g. an
    incomplete/aborted run)."""
    d = df.dropna(subset=[param])
    if d.empty:
        raise ValueError('no rows with a value for %s' % param)

    colors = dict(ARM_COLORS, **cfg.get('colors', {}))
    fig, axes = plt.subplots(3, 1, figsize=cfg.get('figsize', (8, 11)), sharex=True)
    any_plotted = False

    for arm in ARM_ORDER:
        sub = d[d['trial_mode'] == arm]
        if sub.empty:
            continue
        color = colors.get(arm)
        any_plotted = True

        g_dec = sub.groupby(param)['time_to_decision_s'].agg(['mean', 'std']).sort_index()
        axes[0].errorbar(g_dec.index, g_dec['mean'], yerr=g_dec['std'].fillna(0.0),
                         marker='o', capsize=3, color=color, label=arm)

        g_full = sub.groupby(param)['t_probe_to_classify_exit_s'].agg(['mean', 'std']).sort_index()
        axes[1].errorbar(g_full.index, g_full['mean'], yerr=g_full['std'].fillna(0.0),
                         marker='o', capsize=3, color=color, label=arm)

        expected = _expected_label(arm)
        correct = (sub['classified_label'] == expected).astype(float)
        g_acc = correct.groupby(sub[param]).mean().sort_index()
        axes[2].plot(g_acc.index, g_acc.values, marker='o', color=color, label=arm)

    if not any_plotted:
        plt.close(fig)
        raise ValueError('no movable/solid rows for %s' % param)

    legend_fontsize = cfg.get('legend_fontsize', 8)
    axes[0].set_ylabel(cfg.get('decision_ylabel', 'time (s)'))
    axes[0].set_title(cfg.get('decision_title',
                              'Time to classification DECISION (classify_time window itself)'))
    axes[1].set_ylabel(cfg.get('full_ylabel', 'time (s)'))
    axes[1].set_title(cfg.get('full_title', 'Time PROBE-entry to DECISION\n'
                              '(the "classification time" this sweep tunes)'),
                      fontsize=cfg.get('full_title_fontsize', 10))
    axes[2].set_ylabel(cfg.get('accuracy_ylabel', 'fraction correctly classified'))
    axes[2].set_title(cfg.get('accuracy_title', 'Classification correctness vs. expected label'))
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel(cfg.get('xlabel', PARAM_LABELS.get(param, param)))
    for ax in axes:
        ax.legend(fontsize=legend_fontsize)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
    plt.close(fig)


def plot_diagnostics_vs_param(df, param, out_path, cfg):
    """Stability (peak roll/pitch) and wrench-latency vs. one tunable
    parameter -- the "is this value actually safe/healthy" complement to
    plot_time_vs_param's pure timing curves."""
    d = df.dropna(subset=[param])
    if d.empty:
        raise ValueError('no rows with a value for %s' % param)

    colors = dict(ARM_COLORS, **cfg.get('colors', {}))
    fig, axes = plt.subplots(2, 1, figsize=cfg.get('figsize', (7, 8)), sharex=True)
    any_plotted = False

    for arm in ARM_ORDER:
        sub = d[d['trial_mode'] == arm]
        if sub.empty:
            continue
        color = colors.get(arm)
        any_plotted = True

        peak = sub[['peak_roll_deg', 'peak_pitch_deg']].max(axis=1)
        g_peak = peak.groupby(sub[param]).agg(['mean', 'std']).sort_index()
        axes[0].errorbar(g_peak.index, g_peak['mean'], yerr=g_peak['std'].fillna(0.0),
                         marker='o', capsize=3, color=color, label=arm)

        g_lat = sub.groupby(param)['cpm_wrench_latency_ms_mean'].agg(['mean', 'std']).sort_index()
        axes[1].errorbar(g_lat.index, g_lat['mean'], yerr=g_lat['std'].fillna(0.0),
                         marker='o', capsize=3, color=color, label=arm)

    if not any_plotted:
        plt.close(fig)
        raise ValueError('no movable/solid rows for %s' % param)

    legend_fontsize = cfg.get('legend_fontsize', 8)
    axes[0].set_ylabel(cfg.get('stability_ylabel', 'peak |roll or pitch| (deg)'))
    axes[0].set_title(cfg.get('stability_title', 'Base stability during the trial'))
    axes[1].set_ylabel(cfg.get('latency_ylabel', 'mean wrench latency (ms)'))
    axes[1].set_title(cfg.get('latency_title', 'contact_probe_manager wrench latency'))
    axes[1].set_xlabel(cfg.get('xlabel', PARAM_LABELS.get(param, param)))
    for ax in axes:
        ax.legend(fontsize=legend_fontsize)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
    plt.close(fig)


def plot_diagnostics_over_sweep(df, out_path, cfg):
    """Whole-sweep sanity check: did contact_probe_manager's loop rate /
    wrench latency stay stable trial to trial, independent of which
    parameter was being varied?"""
    d = df.sort_values('trial_id')
    target_hz = cfg.get('target_loop_rate_hz', 10.0)
    fig, axes = plt.subplots(2, 1, figsize=cfg.get('figsize', (8, 6)), sharex=True)
    axes[0].plot(d['trial_id'], d['cpm_loop_rate_hz_mean'], marker='o', ms=3,
                c=cfg.get('loop_rate_color', 'tab:blue'))
    axes[0].axhline(target_hz, ls='--', c='gray', lw=1, label='target (%.0f Hz)' % target_hz)
    axes[0].set_ylabel('loop rate (Hz)')
    axes[0].legend(fontsize=cfg.get('legend_fontsize', 8))
    axes[0].grid(alpha=0.3)
    axes[1].plot(d['trial_id'], d['cpm_wrench_latency_ms_mean'], marker='o', ms=3,
                c=cfg.get('latency_color', 'tab:orange'))
    axes[1].set_ylabel('wrench latency (ms)')
    axes[1].set_xlabel('trial_id')
    axes[1].grid(alpha=0.3)
    fig.suptitle(cfg.get('title', 'contact_probe_manager runtime diagnostics over the tuning sweep'))
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
titles/axis labels/legend text/colors/figure size, then re-run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tuning_plotting import generate_all

if __name__ == '__main__':
    tag_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    written, errors = generate_all(tag_dir)
    for path in written:
        print('wrote', path)
    for filename, e in errors:
        print('SKIPPED %s: %s' % (filename, e), file=sys.stderr)
'''


def generate_all(tag_dir):
    """Writes every plot into <tag_dir>/plots/, per-plot config into
    <tag_dir>/plots/config/, and a self-contained script bundle into
    <tag_dir>/plot_scripts/ (see module docstring). Returns (written,
    errors) where written is a list of file paths and errors is a list of
    (filename, exception) tuples -- one bad/missing column skips only that
    plot, mirroring plotting.py's generate_all()."""
    config = load_or_init_config(tag_dir, default_config())
    df = _load_summary(tag_dir, config)
    plots_dir = os.path.join(tag_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    written, errors = [], []

    def _run(filename, fn, *args):
        out_path = os.path.join(plots_dir, filename)
        section = os.path.splitext(filename)[0]
        try:
            fn(df, *args, out_path, config[section])
            written.append(out_path)
        except Exception as e:
            errors.append((filename, e))

    for param in SWEEP_PARAMS:
        _run('time_vs_%s.png' % param, plot_time_vs_param, param)
        _run('diagnostics_vs_%s.png' % param, plot_diagnostics_vs_param, param)
    _run('diagnostics_over_sweep.png', plot_diagnostics_over_sweep)

    bundle_scripts(tag_dir, [_THIS_FILE, _PLOT_CONFIG_FILE], _REGENERATE_SCRIPT)

    return written, errors
