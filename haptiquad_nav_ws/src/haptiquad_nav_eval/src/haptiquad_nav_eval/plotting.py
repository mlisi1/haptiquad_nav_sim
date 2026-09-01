"""
plotting.py — generates plots from a completed (or in-progress)
haptiquad_nav_eval results directory (a <results_dir>/<tag>/ folder
containing summary.csv and its trial_NNNNN_timeseries.csv files).

Pure post-processing over the CSVs trial_logger.py writes -- no ROS runtime
required (no rospy import anywhere in this module), so it runs standalone
at any time against any past results, not just automatically at the end of
a live sweep. See generate_all(), the single entry point
sweep_orchestrator.py calls when a sweep completes, and
scripts/plot_results.py, the CLI wrapper for running it by hand.

Config-driven, one YAML file per plot: generate_all() writes (or, on a
re-run, reads back and reuses) <tag_dir>/plots/config/<plot_name>.yaml for
every plot, plus a shared config/ground_truth.yaml for the reference values
several plots share. Edit any of those files and re-run to change just that
plot's parameters -- no simulation, no ROS, no rebuild required. See
DEFAULT_CONFIG below for every tunable and its default.

generate_all() also copies this module plus a tiny standalone runner into
<tag_dir>/plot_scripts/, so the results folder is self-contained: the
plotting code and its config travel with the data, and `python3
plot_scripts/regenerate.py` works from just that folder (pandas, numpy,
matplotlib, pyyaml -- no ROS, no rospkg) even if the haptiquad_nav_eval
package isn't checked out anywhere nearby.

Each plot function takes (df, out_path, cfg) and is wrapped individually in
try/except by generate_all() so one bad/missing column (e.g. safety fields
recorded before trial_logger.py added them -- see that module's docstring)
skips only that one plot instead of aborting the rest.
"""
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless -- this runs inside the sim container, no display
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import yaml
except ImportError:  # pragma: no cover -- normally present in this ROS env
    yaml = None


G = 9.81  # m/s^2, matches trial_core.G / default.yaml's gravity_z

LABEL_COLORS = {
    'MOVABLE': 'tab:green',
    'SOLID': 'tab:red',
    'NO_CONTACT': 'tab:gray',
    'INCOMPLETE': 'black',
}
STATE_SHADE_COLORS = {
    'SLOW_APPROACH': 'tab:cyan',
    'CONTACT_PROBE': 'tab:blue',
    'CLASSIFYING': 'tab:orange',
    'BACKING_UP': 'tab:purple',
}
STATE_SHADE_DESCRIPTIONS = {
    'SLOW_APPROACH': 'SLOW_APPROACH (speed capped, still approaching)',
    'CONTACT_PROBE': 'CONTACT_PROBE (probing, waiting for sustained contact)',
    'CLASSIFYING': 'CLASSIFYING (fixed-duration push, deciding movable/solid)',
    'BACKING_UP': 'BACKING_UP (reversing clear of a confirmed solid)',
}


def _label_color(label):
    return LABEL_COLORS.get(label, 'tab:brown')


def _require_columns(df, cols, hint):
    """Raise a clear, specific error (instead of a raw pandas KeyError) if
    any of `cols` are entirely absent from df -- the common case being a
    results directory recorded before trial_logger.py added that
    instrumentation and hasn't been restarted since."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError('missing column(s) %s -- %s' % (missing, hint))


# ── Default configuration ────────────────────────────────────────────────────
# One key per plot (matching its output filename stem), plus a shared
# 'ground_truth' section several plots read from. Written out as one YAML
# file per key by generate_all() -- see that function and the module
# docstring. NOTHING in this dict is read directly by the plot functions at
# call time except via the `cfg` argument they're passed, so editing a
# written-out YAML and re-running always takes effect.

def _live_probe_threshold_fx(default=150.0):
    """Best-effort read of contact_probe.yaml's live wrench_threshold/fx,
    used ONLY to seed ground_truth.yaml the first time a tag directory's
    config doesn't exist yet. Never called again after that -- subsequent
    runs (including the bundled plot_scripts/ copy, which may not even be
    inside a ROS workspace) read the already-written YAML instead. Safe to
    call with no live ROS system running: this reads contact_probe.yaml as
    a plain file via rospkg's package-path lookup, no topics/services/nodes
    touched."""
    try:
        import rospkg
        pkg_path = rospkg.RosPack().get_path('haptiquad_nav')
        with open(os.path.join(pkg_path, 'config', 'contact_probe.yaml')) as f:
            cfg = yaml.safe_load(f)
        return float(cfg['wrench_threshold']['fx'])
    except Exception:
        return default


def _live_backup_min_clearance(default=0.60):
    """Same idea as _live_probe_threshold_fx(), for backup_min_clearance."""
    try:
        import rospkg
        pkg_path = rospkg.RosPack().get_path('haptiquad_nav')
        with open(os.path.join(pkg_path, 'config', 'contact_probe.yaml')) as f:
            cfg = yaml.safe_load(f)
        return float(cfg['backup_min_clearance'])
    except Exception:
        return default


def default_config():
    """Fresh copy of every plot's default parameters. Robot mass: sum of all
    59 link masses in anymal_c_simple_description/urdf/anymal.urdf (52.13 kg
    -- matches ANYmal C's published ~50-55 kg spec). push_capability_mass_fraction
    is a documented ASSUMPTION, not a measurement: the theoretical Coulomb
    traction limit at this sim's ground friction (mu=1.0 in the test worlds)
    is the full body weight, which is unrealistic for a walking (not braced)
    push since only 2-3 feet are in stance at once and the robot must
    preserve balance rather than lean fully into the push. Override if you
    have a better number (e.g. from an actual push-to-failure test)."""
    return {
        'ground_truth': {
            'threshold_fx_n': _live_probe_threshold_fx(),
            'robot_mass_kg': 52.13,
            'push_capability_mass_fraction': 0.25,
        },
        'classification_scatter': {
            'show_gt_boundaries': True,
        },
        'confusion_matrix_probe_threshold': {},
        'confusion_matrix_mass_fraction': {},
        'force_vs_static_friction': {},
        'timing_by_label': {},
        'diagnostics': {
            'loop_rate_target_hz': 10.0,
        },
        'timeseries_examples': {
            'n_per_label': 2,
        },
        'stability': {
            # Reference line only, not a hard safety cutoff enforced anywhere
            # in the running system -- pick a value you actually trust and
            # the plot will show how close the sweep came to it.
            'roll_pitch_safe_limit_deg': 15.0,
        },
        'safety_margin_backup': {
            'backup_min_clearance_m': _live_backup_min_clearance(),
        },
        'estimator_health': {},
        'correlations': {},
    }


def _load_or_init_config(tag_dir):
    """Load <tag_dir>/plots/config/*.yaml if present (so hand-edits survive
    a re-run); otherwise seed from default_config() and write it out. Missing
    individual files/keys fall back to defaults rather than erroring, so
    adding a new plot later doesn't require regenerating old config dirs."""
    config_dir = os.path.join(tag_dir, 'plots', 'config')
    os.makedirs(config_dir, exist_ok=True)
    defaults = default_config()
    resolved = {}
    for name, default_section in defaults.items():
        path = os.path.join(config_dir, name + '.yaml')
        if os.path.exists(path) and yaml is not None:
            try:
                with open(path) as f:
                    loaded = yaml.safe_load(f) or {}
                section = dict(default_section)
                section.update(loaded)
            except Exception:
                section = dict(default_section)
        else:
            section = dict(default_section)
        resolved[name] = section
        if yaml is not None:
            with open(path, 'w') as f:
                yaml.safe_dump(section, f, sort_keys=False)
    return resolved


# ── Data loading ─────────────────────────────────────────────────────────────

def load_summary(tag_dir, config):
    df = pd.read_csv(os.path.join(tag_dir, 'summary.csv'))
    gt = config['ground_truth']
    df['static_force_n'] = df['mu'] * df['mass_kg'] * G
    df['peak_force_horizontal'] = np.sqrt(df['peak_fx']**2 + df['peak_fy']**2)
    df['gt_movable_threshold'] = df['static_force_n'] < gt['threshold_fx_n']
    push_capability_n = gt['push_capability_mass_fraction'] * gt['robot_mass_kg'] * G
    df['gt_movable_mass_fraction'] = df['static_force_n'] < push_capability_n
    return df


# ── Individual plots ─────────────────────────────────────────────────────────

def plot_classification_scatter(df, out_path, cfg, gt_cfg):
    """The classifier's empirical decision boundary across the tested
    (mass, mu) grid, overlaid with the two ground-truth boundary curves
    (mu = reference_force / (mass * g)). This is what makes the scatter
    more than a duplicate of the confusion matrices: it shows WHERE in
    parameter space the classifier's behavior diverges from each ground
    truth, not just how often."""
    threshold_fx_n = gt_cfg['threshold_fx_n']
    push_capability_n = gt_cfg['push_capability_mass_fraction'] * gt_cfg['robot_mass_kg'] * G

    geoms = sorted(df['geometry'].dropna().unique()) or [None]
    fig, axes = plt.subplots(1, len(geoms), figsize=(6 * len(geoms), 5), squeeze=False)
    mass_lo = max(df['mass_kg'].min(skipna=True) * 0.9, 0.1)
    mass_hi = df['mass_kg'].max(skipna=True) * 1.05
    mass_curve = np.linspace(mass_lo, mass_hi, 200)

    for ax, geom in zip(axes[0], geoms):
        sub = df if geom is None else df[df['geometry'] == geom]
        for label, group in sub.groupby('classified_label'):
            ax.scatter(group['mass_kg'], group['mu'], c=_label_color(label), label=label,
                      s=70, edgecolors='k', linewidths=0.5, zorder=3)

        if cfg.get('show_gt_boundaries', True):
            ax.plot(mass_curve, threshold_fx_n / (mass_curve * G), c='tab:purple', ls='--', lw=1.5,
                   label='GT boundary: probe threshold (%.0f N)' % threshold_fx_n, zorder=2)
            ax.plot(mass_curve, push_capability_n / (mass_curve * G), c='tab:olive', ls=':', lw=1.5,
                   label='GT boundary: %.0f%% robot weight (%.0f N)'
                         % (gt_cfg['push_capability_mass_fraction'] * 100, push_capability_n), zorder=2)

        ax.set_xlabel('mass (kg)')
        ax.set_ylabel('friction coefficient (mu)')
        ax.set_ylim(bottom=0)
        ax.set_title(geom or 'all geometries')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, zorder=0)
    fig.suptitle('Classification outcome vs mass / friction, with ground-truth boundaries')
    fig.tight_layout(rect=(0, 0, 1, 0.92))  # reserve headroom so suptitle doesn't collide
    fig.savefig(out_path, dpi=150)          # with a subplot title when there's only 1 geometry
    plt.close(fig)


def plot_force_vs_static_friction(df, out_path, cfg):
    """Sanity check on the force ESTIMATOR (not the classification decision):
    does measured peak contact force scale sensibly with the physically-
    expected static friction force (mu * mass * g)? Values tracking or
    exceeding the y=x reference are expected (quasi-static pushing plus
    impact/dynamic effects), and this also explains why the wrench
    threshold sits where it does relative to the forces actually being
    generated in this sweep."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group in df.groupby('classified_label'):
        ax.scatter(group['static_force_n'], group['peak_force_horizontal'],
                  c=_label_color(label), label=label, s=60, edgecolors='k',
                  linewidths=0.5, zorder=3)
    hi = max(df['static_force_n'].max(skipna=True),
             df['peak_force_horizontal'].max(skipna=True), 1.0) * 1.05
    ax.plot([0, hi], [0, hi], ls='--', c='gray', lw=1, label='y = x (reference)', zorder=2)
    ax.set_xlabel('static friction force  mu * mass * g  [N]')
    ax.set_ylabel('peak measured horizontal force  sqrt(fx^2+fy^2)  [N]')
    ax.set_title('Measured contact force vs expected static friction')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _confusion_counts(df, gt_col):
    """Positive class = MOVABLE. Only rows the classifier actually reached a
    MOVABLE/SOLID verdict on are scored -- NO_CONTACT/INCOMPLETE trials
    never got a classification attempt, so they're excluded (n_excluded)
    rather than silently counted as errors either way."""
    attempted = df[df['classified_label'].isin(['MOVABLE', 'SOLID'])].dropna(subset=[gt_col])
    pred_movable = attempted['classified_label'] == 'MOVABLE'
    gt_movable = attempted[gt_col].astype(bool)
    tp = int((pred_movable & gt_movable).sum())
    fn = int((~pred_movable & gt_movable).sum())
    fp = int((pred_movable & ~gt_movable).sum())
    tn = int((~pred_movable & ~gt_movable).sum())
    n_excluded = len(df) - len(attempted)
    return tp, fn, fp, tn, n_excluded


def plot_confusion_matrix(df, out_path, cfg, gt_col, gt_description):
    """Classifier's MOVABLE/SOLID verdict vs one ground-truth definition.
    Rows = ground truth, columns = classifier prediction (sklearn
    convention). See gt_movable_threshold / gt_movable_mass_fraction in
    load_summary() for what gt_col means."""
    tp, fn, fp, tn, n_excluded = _confusion_counts(df, gt_col)
    mat = np.array([[tp, fn], [fp, tn]])
    total = tp + fn + fp + tn

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mat, cmap='Blues', vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha='center', va='center', fontsize=18,
                   color='white' if mat[i, j] > mat.max() / 2 else 'black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['pred MOVABLE', 'pred SOLID'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['GT MOVABLE', 'GT SOLID'])
    ax.set_xlabel('classifier verdict')
    ax.set_ylabel('ground truth')

    if total > 0:
        acc = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) else float('nan')
        recall = tp / (tp + fn) if (tp + fn) else float('nan')
        stats = 'accuracy=%.2f  precision=%.2f  recall=%.2f  (n=%d' % (acc, precision, recall, total)
    else:
        stats = '(n=0'
    stats += ', %d NO_CONTACT/INCOMPLETE excluded)' % n_excluded if n_excluded else ')'
    ax.set_title('%s\n%s' % (gt_description, stats), fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _boxplot_by_label(ax, df, value_col, title):
    present = [l for l in df['classified_label'].dropna().unique()]
    ordered = [l for l in LABEL_COLORS if l in present] + [l for l in present if l not in LABEL_COLORS]
    data = [df.loc[df['classified_label'] == l, value_col].dropna().values for l in ordered]
    ordered = [l for l, d in zip(ordered, data) if len(d) > 0]
    data = [d for d in data if len(d) > 0]
    if not data:
        ax.set_title(title + '\n(no data)')
        return False
    bp = ax.boxplot(data, labels=ordered, patch_artist=True)
    for patch, label in zip(bp['boxes'], ordered):
        patch.set_facecolor(_label_color(label))
        patch.set_alpha(0.5)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    return True


def plot_timing_by_label(df, out_path, cfg):
    """Two panels, deliberately separated because they answer different
    questions and conflating them is misleading:

    Left  -- time to DECISION (t_classify_entry -> t_classify_exit, i.e.
             classify_time). This is a fixed-duration window applied
             identically regardless of outcome, so it should be about the
             SAME for MOVABLE and SOLID -- there is no real "movable is
             faster to detect" effect; this panel exists to demonstrate that.
    Right -- time to full RESOLUTION (nav_time_to_classification: goal-sent
             to state machine back in NORMAL_NAV). This one IS asymmetric,
             but for a structural reason: a SOLID verdict additionally
             requires the whole BACKING_UP maneuver before the state
             machine resolves, which a MOVABLE verdict skips entirely. The
             asymmetry lives here, not in the classification decision.
    """
    df = df.copy()
    df['time_to_decision_s'] = df['t_classify_exit'] - df['t_classify_entry']

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ok1 = _boxplot_by_label(axes[0], df, 'time_to_decision_s',
                            'Time to classification DECISION\n(classify_time window -- same for both, by design)')
    axes[0].set_ylabel('time (s)')
    ok2 = _boxplot_by_label(axes[1], df, 'nav_time_to_classification',
                            'Time to full state-machine RESOLUTION\n(SOLID includes BACKING_UP; MOVABLE does not)')
    axes[1].set_ylabel('time (s)')
    if not (ok1 or ok2):
        raise ValueError('no timing data to plot')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_diagnostics(df, out_path, cfg):
    """Node-health sanity check over the sweep: did contact_probe_manager's
    loop rate / wrench latency stay stable trial to trial?"""
    target_hz = cfg.get('loop_rate_target_hz', 10.0)
    d = df.sort_values('trial_id')
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(d['trial_id'], d['cpm_loop_rate_hz_mean'], marker='o', ms=3)
    axes[0].axhline(target_hz, ls='--', c='gray', lw=1, label='target (%.0f Hz)' % target_hz)
    axes[0].set_ylabel('loop rate (Hz)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    # Sim time is deterministic (fixed real_time_factor), so this is usually
    # ~10.0000 Hz with only 4th/5th-decimal jitter -- matplotlib's default
    # scientific offset notation (e.g. "+1e1") makes that unreadable. Show
    # plain values on a fixed window around the target instead.
    axes[0].ticklabel_format(axis='y', useOffset=False, style='plain')
    lo = min(d['cpm_loop_rate_hz_mean'].min(skipna=True), target_hz - 0.1)
    hi = max(d['cpm_loop_rate_hz_mean'].max(skipna=True), target_hz + 0.1)
    pad = max((hi - lo) * 0.1, 0.01)
    axes[0].set_ylim(lo - pad, hi + pad)
    axes[1].plot(d['trial_id'], d['cpm_wrench_latency_ms_mean'], marker='o', ms=3, c='tab:orange')
    axes[1].set_ylabel('wrench latency (ms)')
    axes[1].set_xlabel('trial_id')
    axes[1].grid(alpha=0.3)
    fig.suptitle('contact_probe_manager runtime diagnostics over the sweep')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pick_examples(df, n_per_label):
    picked = []
    for _label, group in df.groupby('classified_label'):
        picked.extend(group.head(n_per_label)['trial_id'].tolist())
    return sorted(set(picked))


def plot_timeseries_examples(df, out_path, cfg):
    """Detailed per-trial inspection: |F| and forward displacement over
    time, with contact-probe state segments shaded, for a handful of
    representative trials spanning every classified_label present."""
    n_per_label = cfg.get('n_per_label', 2)
    trial_ids = _pick_examples(df, n_per_label)
    if not trial_ids:
        raise ValueError('no trials to plot')

    fig, axes = plt.subplots(len(trial_ids), 1, figsize=(9, 2.8 * len(trial_ids) + 0.6), squeeze=False)
    axes = axes[:, 0]
    seen_states = {}   # state -> Patch, accumulated across all rows, dedup'd, in STATE_SHADE_COLORS order
    line_handles, line_labels = [], []
    for ax, tid in zip(axes, trial_ids):
        row = df[df['trial_id'] == tid].iloc[0]
        ts_path = row['timeseries_csv_path']
        if not os.path.exists(ts_path):
            ax.set_title('trial %d: timeseries file missing (%s)' % (tid, ts_path))
            ax.axis('off')
            continue
        ts = pd.read_csv(ts_path)
        if ts.empty:
            ax.set_title('trial %d: empty timeseries (never entered CONTACT_PROBE)' % tid)
            ax.axis('off')
            continue

        for state, color in STATE_SHADE_COLORS.items():
            mask = (ts['state'] == state).values
            if mask.any():
                ax.fill_between(ts['t_since_probe_entry'], 0, 1, where=mask,
                               transform=ax.get_xaxis_transform(), color=color,
                               alpha=0.15, step='post', zorder=0)
                if state not in seen_states:
                    seen_states[state] = Patch(facecolor=color, alpha=0.3,
                                              label=STATE_SHADE_DESCRIPTIONS[state])

        # Raw |F| is dominated by gait-cycle noise (see CLAUDE.md's notes on
        # MomentumObserver walking-dynamics noise) -- plot it faint as
        # context, and a rolling-mean overlay as the readable signal. This
        # window is illustrative only, NOT a reproduction of
        # contact_probe_manager's own per-axis wrench_filter_window
        # averaging (it averages fx/fy/fz/tx/ty/tz separately against
        # separate thresholds, not this combined magnitude).
        fmag = np.sqrt(ts['fx']**2 + ts['fy']**2 + ts['fz']**2)
        fmag_smooth = fmag.rolling(50, min_periods=1, center=True).mean()
        ax.plot(ts['t_since_probe_entry'], fmag, c='k', lw=0.5, alpha=0.3, zorder=2)
        l1, = ax.plot(ts['t_since_probe_entry'], fmag_smooth, c='crimson', lw=1.5,
                      label='|F| (N), smoothed', zorder=3)
        ax2 = ax.twinx()
        l2, = ax2.plot(ts['t_since_probe_entry'], ts['forward_disp_since_probe'],
                       c='tab:blue', lw=1, ls='--', label='forward disp (m)', zorder=3)
        if not line_handles:
            line_handles, line_labels = [l1, l2], [l1.get_label(), l2.get_label()]

        ax.set_title('trial %d -- %s (mass=%.1fkg, mu=%.2f)'
                    % (tid, row['classified_label'], row['mass_kg'], row['mu']))
        ax.set_ylabel('|F| (N)')
        ax2.set_ylabel('fwd disp (m)', color='tab:blue')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('t since CONTACT_PROBE entry (s)')
    # One shared legend for the whole figure -- the state-color meaning and
    # line styles are identical across every row, so repeating a 6-entry
    # legend per subplot would just be clutter.
    handles = line_handles + [seen_states[s] for s in STATE_SHADE_COLORS if s in seen_states]
    labels = line_labels + [h.get_label() for h in handles[len(line_handles):]]
    if handles:
        fig.legend(handles, labels, fontsize=8, loc='upper center',
                  ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93 if len(trial_ids) <= 2 else 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Safety / stability plots ─────────────────────────────────────────────────
# See trial_logger.py's module docstring for what's actually gathered and
# what it does NOT prove (sim behavior, not a real-hardware safety
# certification). These three plots + the correlations below are the
# direct output of that instrumentation.

def plot_stability(df, out_path, cfg):
    """Peak roll/pitch and peak roll/pitch RATE per trial, against a
    reference "safe" line (a plotting aid, not an enforced limit anywhere
    in the running system -- set roll_pitch_safe_limit_deg in
    config/stability.yaml to a value you trust)."""
    safe_deg = cfg.get('roll_pitch_safe_limit_deg', 15.0)
    needed = ['peak_roll_deg', 'peak_pitch_deg', 'peak_roll_rate_dps', 'peak_pitch_rate_dps']
    _require_columns(df, needed, 'trial_logger.py needs restarting after the roll/pitch '
                                 'instrumentation was added (see its module docstring)')
    if df[needed].isna().all().all():
        raise ValueError('stability columns are present but entirely NaN for every trial in '
                         'this results directory -- recorded before trial_logger.py was restarted')

    d = df.sort_values('trial_id')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, col, ylabel, safe_line in (
            (axes[0, 0], 'peak_roll_deg', 'peak |roll| (deg)', safe_deg),
            (axes[0, 1], 'peak_pitch_deg', 'peak |pitch| (deg)', safe_deg),
            (axes[1, 0], 'peak_roll_rate_dps', 'peak |roll rate| (deg/s)', None),
            (axes[1, 1], 'peak_pitch_rate_dps', 'peak |pitch rate| (deg/s)', None)):
        for label, group in d.groupby('classified_label'):
            ax.scatter(group['trial_id'], group[col], c=_label_color(label), label=label,
                      s=40, edgecolors='k', linewidths=0.4, zorder=3)
        if safe_line is not None:
            ax.axhline(safe_line, ls='--', c='red', lw=1, alpha=0.6,
                      label='reference limit (%.0f deg)' % safe_line, zorder=2)
        ax.set_xlabel('trial_id')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle('Base stability (roll/pitch) across the sweep')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_safety_margin_backup(df, out_path, cfg):
    """How much clearance the robot actually kept from an unknown rear
    obstacle while reversing (BACKING_UP), vs the configured safety minimum
    (contact_probe.yaml's backup_min_clearance). Only SOLID trials ever
    enter BACKING_UP, so this is empty for a sweep with no SOLID outcomes --
    that's expected, not a bug."""
    configured_min = cfg.get('backup_min_clearance_m', 0.60)
    _require_columns(df, ['min_rear_clearance_during_backup_m'],
                     'contact_probe_manager needs restarting for the nearest-obstacle '
                     'publishers to exist (see trial_logger.py\'s module docstring)')
    sub = df[df['classified_label'] == 'SOLID'].dropna(subset=['min_rear_clearance_during_backup_m'])
    if sub.empty:
        raise ValueError('no BACKING_UP clearance data (either no SOLID trials, or '
                         'contact_probe_manager needs restarting for the nearest-obstacle '
                         'publishers to exist -- see trial_logger.py\'s module docstring)')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub['trial_id'], sub['min_rear_clearance_during_backup_m'],
              c='tab:red', s=50, edgecolors='k', linewidths=0.5, zorder=3,
              label='min clearance actually kept')
    ax.axhline(configured_min, ls='--', c='black', lw=1.5,
              label='configured backup_min_clearance (%.2f m)' % configured_min, zorder=2)
    violated = sub[sub['min_rear_clearance_during_backup_m'] < configured_min]
    if not violated.empty:
        ax.scatter(violated['trial_id'], violated['min_rear_clearance_during_backup_m'],
                  facecolors='none', edgecolors='red', s=180, linewidths=2, zorder=4,
                  label='below configured minimum (%d trials)' % len(violated))
    ax.set_xlabel('trial_id')
    ax.set_ylabel('min clearance to unknown rear obstacle during BACKING_UP (m)')
    ax.set_title('Backup safety margin vs configured minimum')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_estimator_health(df, out_path, cfg):
    """HaptiQuad's own MomentumObserver residual estimation error
    (/residual_errors) during each trial -- a proxy for how much to trust
    the wrench signal the classifier acts on, independent of what that
    signal actually reads."""
    needed = ['peak_residual_err_int', 'peak_residual_err_ext']
    _require_columns(df, needed, 'trial_logger.py needs restarting after the '
                                 '/residual_errors subscription was added')
    if df[needed].isna().all().all():
        raise ValueError('residual columns are present but entirely NaN for every trial in '
                         'this results directory -- recorded before trial_logger.py was restarted')

    d = df.sort_values('trial_id')
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, col, ylabel, color in (
            (axes[0], 'peak_residual_err_int', 'peak |internal residual error|', 'tab:purple'),
            (axes[1], 'peak_residual_err_ext', 'peak |external residual error|', 'tab:brown')):
        for label, group in d.groupby('classified_label'):
            ax.scatter(group['trial_id'], group[col], c=_label_color(label), label=label,
                      s=40, edgecolors='k', linewidths=0.4, zorder=3)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel('trial_id')
    fig.suptitle('HaptiQuad estimator residual error across the sweep\n'
                "(higher = less trustworthy wrench reading at that moment)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlations(df, out_path, cfg):
    """Grid of relationships worth checking by eye: does harder contact
    correlate with more instability or estimator error, and does object
    mass predictably affect classification-window displacement. Purely
    descriptive (scatter + naive linear trend line for visual reference,
    not a fitted/claimed model)."""
    pairs = [
        ('peak_force_horizontal', 'peak_roll_deg',
         'peak horizontal force (N)', 'peak |roll| (deg)',
         'Contact force vs base roll'),
        ('peak_force_horizontal', 'peak_pitch_deg',
         'peak horizontal force (N)', 'peak |pitch| (deg)',
         'Contact force vs base pitch'),
        ('static_force_n', 'peak_residual_err_ext',
         'static friction force mu*mass*g (N)', 'peak external residual error',
         'Contact difficulty vs estimator trustworthiness'),
        ('mass_kg', 'delta_d_classify_window',
         'mass (kg)', 'forward displacement during classify window (m)',
         'Object mass vs classification-window displacement'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    any_ok = False
    for ax, (xcol, ycol, xlabel, ylabel, title) in zip(axes.flat, pairs):
        if xcol not in df.columns or ycol not in df.columns:
            ax.set_title(title + '\n(columns missing)')
            ax.axis('off')
            continue
        sub = df[[xcol, ycol, 'classified_label']].dropna()
        if sub.empty:
            ax.set_title(title + '\n(no data)')
            ax.axis('off')
            continue
        any_ok = True
        for label, group in sub.groupby('classified_label'):
            ax.scatter(group[xcol], group[ycol], c=_label_color(label), label=label,
                      s=40, edgecolors='k', linewidths=0.4, zorder=3)
        if len(sub) >= 2 and sub[xcol].std() > 0:
            m, b = np.polyfit(sub[xcol], sub[ycol], 1)
            xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 50)
            ax.plot(xs, m * xs + b, c='gray', ls='--', lw=1, alpha=0.7, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    if not any_ok:
        raise ValueError('no correlation data available (missing columns / no rows)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────

_THIS_FILE = os.path.abspath(__file__)

_REGENERATE_SCRIPT = '''#!/usr/bin/env python3
"""Standalone regenerator for this results folder's plots. No ROS, no
rospkg needed -- ground-truth reference values were already baked into
../plots/config/*.yaml when this bundle was created. Just:

    python3 regenerate.py

Requires: pandas, numpy, matplotlib, pyyaml (plain pip packages).
Edit the YAML files in ../plots/config/ first to change any plot's
parameters, then re-run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting import generate_all

if __name__ == '__main__':
    tag_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    written, errors = generate_all(tag_dir)
    for path in written:
        print('wrote', path)
    for filename, e in errors:
        print('SKIPPED %s: %s' % (filename, e), file=sys.stderr)
'''


def _bundle_scripts(tag_dir):
    """Copy this module + a tiny runner into <tag_dir>/plot_scripts/, so the
    results folder is self-contained (see module docstring). When
    regenerate.py is run FROM that bundle (the documented, primary way to
    use it), the running plotting.py IS already the bundle's own copy --
    _THIS_FILE and the copy destination are then the same path, which
    shutil.copyfile refuses (SameFileError). That's a no-op, not an error:
    the bundle is already up to date with itself."""
    bundle_dir = os.path.join(tag_dir, 'plot_scripts')
    os.makedirs(bundle_dir, exist_ok=True)
    dest = os.path.join(bundle_dir, 'plotting.py')
    if os.path.abspath(dest) != os.path.abspath(_THIS_FILE):
        shutil.copyfile(_THIS_FILE, dest)
    with open(os.path.join(bundle_dir, 'regenerate.py'), 'w') as f:
        f.write(_REGENERATE_SCRIPT)
    os.chmod(os.path.join(bundle_dir, 'regenerate.py'), 0o755)


def generate_all(tag_dir, n_timeseries_examples_per_label=None):
    """Generate every plot for one results tag directory (a folder
    containing summary.csv). Writes into <tag_dir>/plots/, per-plot config
    into <tag_dir>/plots/config/, and a self-contained script bundle into
    <tag_dir>/plot_scripts/. Returns (written_paths, errors) -- errors is a
    list of (filename, exception) for any plot that failed, so a caller can
    log them without the whole call raising.

    n_timeseries_examples_per_label, if given, overrides whatever is in
    (or would be written to) config/timeseries_examples.yaml -- kept as a
    kwarg for backwards-compatible callers; prefer editing the YAML.
    """
    config = _load_or_init_config(tag_dir)
    if n_timeseries_examples_per_label is not None:
        config['timeseries_examples']['n_per_label'] = n_timeseries_examples_per_label

    df = load_summary(tag_dir, config)
    plots_dir = os.path.join(tag_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    written, errors = [], []

    def _run(filename, fn, *args):
        out_path = os.path.join(plots_dir, filename)
        try:
            fn(df, out_path, *args)
            written.append(out_path)
        except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring
            errors.append((filename, e))

    _run('classification_scatter.png', plot_classification_scatter,
        config['classification_scatter'], config['ground_truth'])
    _run('force_vs_static_friction.png', plot_force_vs_static_friction,
        config['force_vs_static_friction'])
    _run('confusion_matrix_probe_threshold.png', plot_confusion_matrix,
        config['confusion_matrix_probe_threshold'], 'gt_movable_threshold',
        'Ground truth: contact-probe wrench threshold (%.0f N)'
        % config['ground_truth']['threshold_fx_n'])
    _run('confusion_matrix_mass_fraction.png', plot_confusion_matrix,
        config['confusion_matrix_mass_fraction'], 'gt_movable_mass_fraction',
        'Ground truth: %.0f%% of robot weight (%.1f kg robot)'
        % (config['ground_truth']['push_capability_mass_fraction'] * 100,
           config['ground_truth']['robot_mass_kg']))
    _run('timing_by_label.png', plot_timing_by_label, config['timing_by_label'])
    _run('diagnostics_over_sweep.png', plot_diagnostics, config['diagnostics'])
    _run('timeseries_examples.png', plot_timeseries_examples, config['timeseries_examples'])
    _run('stability.png', plot_stability, config['stability'])
    _run('safety_margin_backup.png', plot_safety_margin_backup, config['safety_margin_backup'])
    _run('estimator_health.png', plot_estimator_health, config['estimator_health'])
    _run('correlations.png', plot_correlations, config['correlations'])

    _bundle_scripts(tag_dir)

    return written, errors
