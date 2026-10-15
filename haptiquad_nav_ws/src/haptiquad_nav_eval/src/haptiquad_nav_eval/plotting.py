"""
plotting.py — generates plots from a completed (or in-progress)
haptiquad_nav_eval results directory (a <results_dir>/<tag>/ folder whose
csv/ subfolder contains summary.csv and its trial_NNNNN_timeseries.csv
files -- kept out of the tag root so it stays just csv/, plots/,
plot_scripts/, and the contact_probe_manager config snapshot).

Pure post-processing over the CSVs trial_logger.py writes -- no ROS runtime
required (no rospy import anywhere in this module), so it runs standalone
at any time against any past results, not just automatically at the end of
a live sweep. See generate_all(), the single entry point
sweep_orchestrator.py calls when a sweep completes, and
scripts/plot_results.py, the CLI wrapper for running it by hand.

Config-driven, one YAML file per plot: generate_all() writes (or, on a
re-run, reads back and reuses) <tag_dir>/plots/config/<plot_name>.yaml for
every plot, plus a shared config/ground_truth.yaml for the reference values
several plots share, via the shared plot_config.load_or_init_config() (also
used by timing_plotting.py and tuning_plotting.py -- see that module's
docstring). Edit any of those files and re-run to change just that plot's
parameters -- no simulation, no ROS, no rebuild required. See
DEFAULT_CONFIG below for every tunable and its default.

generate_all() also copies this module plus plot_config.py plus a tiny
standalone runner into <tag_dir>/plot_scripts/, so the results folder is
self-contained: the plotting code and its config travel with the data, and
`python3 plot_scripts/regenerate.py` works from just that folder (pandas,
numpy, matplotlib, pyyaml -- no ROS, no rospkg) even if the
haptiquad_nav_eval package isn't checked out anywhere nearby.

Each plot function takes (df, out_path, cfg) and is wrapped individually in
try/except by generate_all() so one bad/missing column (e.g. safety fields
recorded before trial_logger.py added them -- see that module's docstring)
skips only that one plot instead of aborting the rest.
"""
import os
import warnings

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

try:
    # Normal case: imported as haptiquad_nav_eval.plotting.
    from haptiquad_nav_eval.plot_config import (load_or_init_config, bundle_scripts,
                                                exclude_duration_outliers)
except ImportError:
    # Self-contained plot_scripts/ bundle case (see _bundle_scripts's call
    # in generate_all()): plot_config.py sits flat next to this file with
    # no package wrapper, and regenerate.py has already put that directory
    # on sys.path.
    from plot_config import load_or_init_config, bundle_scripts, exclude_duration_outliers


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

def _live_classification_threshold(default=100.0):
    """Best-effort read of contact_probe.yaml's live
    classification_force_threshold_n -- the settled-force MOVABLE/SOLID
    decision cutoff (force-mode classification), NOT contact_force_threshold_n
    (which only gates CONTACT_PROBE detection and, since the fz/torque-noise
    fix, sits far lower -- e.g. 12N -- than any meaningful movable/solid
    boundary). Used ONLY to seed ground_truth.yaml the first time a tag
    directory's config doesn't exist yet. Never called again after that --
    subsequent runs (including the bundled plot_scripts/ copy, which may not
    even be inside a ROS workspace) read the already-written YAML instead.
    Safe to call with no live ROS system running: this reads
    contact_probe.yaml as a plain file via rospkg's package-path lookup, no
    topics/services/nodes touched."""
    try:
        import rospkg
        pkg_path = rospkg.RosPack().get_path('haptiquad_nav')
        with open(os.path.join(pkg_path, 'config', 'contact_probe.yaml')) as f:
            cfg = yaml.safe_load(f)
        return float(cfg['classification_force_threshold_n'])
    except Exception:
        return default


def _live_backup_min_clearance(default=0.60):
    """Same idea as _live_classification_threshold(), for backup_min_clearance."""
    try:
        import rospkg
        pkg_path = rospkg.RosPack().get_path('haptiquad_nav')
        with open(os.path.join(pkg_path, 'config', 'contact_probe.yaml')) as f:
            cfg = yaml.safe_load(f)
        return float(cfg['backup_min_clearance'])
    except Exception:
        return default


def _live_classification_settle_time(default=0.2):
    """Same idea as _live_classification_threshold(), for
    classification_force_settle_time -- used to offline-reconstruct the
    settled force in plot_force_vs_static_friction."""
    try:
        import rospkg
        pkg_path = rospkg.RosPack().get_path('haptiquad_nav')
        with open(os.path.join(pkg_path, 'config', 'contact_probe.yaml')) as f:
            cfg = yaml.safe_load(f)
        return float(cfg['classification_force_settle_time'])
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
        'outlier_filter': {
            'enabled': True,
            # nav_time_to_classification is the closest analog here to
            # timing_plotting.py's total_time_s / tuning_plotting.py's
            # t_probe_to_classify_exit_s -- see
            # plot_config.exclude_duration_outliers()'s docstring for why
            # this filter exists alongside sim_anomaly (a trial can keep
            # moving, just erratically/far slower than normal, without the
            # online stall detector ever firing).
            'value_col': 'nav_time_to_classification',
            'group_by': ['classified_label'],   # median computed per outcome
            'multiplier': 3.0,
            'min_group_size': 3,
        },
        'ground_truth': {
            # The settled-force classification cutoff (contact_probe.yaml's
            # classification_force_threshold_n), NOT the contact-DETECTION
            # threshold (contact_force_threshold_n) -- see
            # _live_classification_threshold()'s docstring for why those two
            # are different numbers and using the wrong one here silently
            # gives a meaningless ground truth.
            'classification_threshold_n': _live_classification_threshold(),
            'robot_mass_kg': 52.13,
            'push_capability_mass_fraction': 0.25,
            # Ground plane's own friction coefficient (see world file's
            # ground_plane <surface><friction><ode><mu>). ODE combines two
            # contacting surfaces' mu via the minimum, so the theoretical
            # static friction force is min(object mu, ground_mu) * mass * g,
            # not just the object's own mu -- matters only if ground_mu is
            # ever lowered below the object mu range tested (currently
            # ground_mu=1.0 is always >= every swept object mu, so it's not
            # the limiting surface in practice, but this keeps the formula
            # correct if that changes).
            'ground_mu': 1.0,
            # Used by force_vs_static_friction to reconstruct the settled
            # force from the raw timeseries -- see classification_force_settle_time
            # in contact_probe.yaml.
            'classification_settle_time_s': _live_classification_settle_time(),
        },
        'classification_scatter': {
            'show_gt_boundaries': True,
            # y-axis cap for friction coefficient -- see the plot's own
            # comment on why the GT boundary curves blow up at low mass.
            # null = auto (tested mu range * mu_axis_margin).
            'mu_axis_max': None,
            'mu_axis_margin': 1.3,
            'figsize_per_geom': [6, 5],   # [width, height] per geometry subplot
            'marker_size': 70,
            'legend_fontsize': 7,
            'dpi': 150,
            'title': 'Classification outcome vs mass / friction, with ground-truth boundaries',
            # Per-label color override, e.g. {MOVABLE: '#2ca02c'} -- merged
            # over the built-in LABEL_COLORS (MOVABLE/SOLID/NO_CONTACT/INCOMPLETE).
            'colors': {},
        },
        'confusion_matrix_force_threshold': {
            'figsize': [5.5, 5],
            'cmap': 'Blues',
            'cell_fontsize': 18,
            'title_fontsize': 10,
            'dpi': 150,
        },
        'confusion_matrix_mass_fraction': {
            'figsize': [5.5, 5],
            'cmap': 'Blues',
            'cell_fontsize': 18,
            'title_fontsize': 10,
            'dpi': 150,
        },
        'force_vs_static_friction': {
            'figsize': [7, 5],
            'marker_size': 60,
            'legend_fontsize': 8,
            'dpi': 150,
            'title': 'Settled contact force vs expected friction',
            'colors': {},
        },
        'timing_by_label': {
            'figsize': [11, 5],
            'dpi': 150,
            'colors': {},
        },
        'diagnostics': {
            'loop_rate_target_hz': 10.0,
            'figsize': [8, 6],
            'marker_size': 3,
            'legend_fontsize': 8,
            'loop_rate_color': 'tab:blue',
            'latency_color': 'tab:orange',
            'dpi': 150,
            'title': 'contact_probe_manager runtime diagnostics over the sweep',
        },
        'timeseries_examples': {
            'n_per_label': 2,
            'row_height': 2.8,          # figure height added per trial row
            'rolling_window': 50,       # samples, for the |F| smoothing overlay
            'force_color': 'crimson',
            'displacement_color': 'tab:blue',
            'legend_fontsize': 8,
            'dpi': 150,
            # Per-state shade color override, merged over the built-in
            # STATE_SHADE_COLORS (SLOW_APPROACH/CONTACT_PROBE/CLASSIFYING/BACKING_UP).
            'state_colors': {},
        },
        'stability': {
            # Reference line only, not a hard safety cutoff enforced anywhere
            # in the running system -- pick a value you actually trust and
            # the plot will show how close the sweep came to it.
            'roll_pitch_safe_limit_deg': 15.0,
            'figsize': [11, 8],
            'marker_size': 40,
            'legend_fontsize': 6,
            'dpi': 150,
            'title': 'Base stability (roll/pitch) across the sweep',
            'colors': {},
        },
        'stability_baseline': {
            'figsize': [11, 8],
            'dpi': 150,
            'colors': ['tab:gray', 'tab:orange'],   # [baseline, contact]
            'title': 'Stability during contact vs normal-walking baseline\n'
                     '(per-trial std within each phase, pooled across the sweep)',
        },
        'stability_vs_param': {
            # plot_stability() plots peak roll/pitch against trial_id (run
            # order, physically meaningless); this plots the same peak
            # values against the sweep's own independent variables instead.
            'roll_pitch_safe_limit_deg': 15.0,
            'figsize': [11, 8],
            'marker_size': 40,
            'legend_fontsize': 7,
            'dpi': 150,
            'title': "Base stability vs. the sweep's own independent variables (mass, friction)",
            'colors': {},
        },
        'stability_distribution': {
            # Distribution view (histogram) of the same peak_roll_deg/
            # peak_pitch_deg columns plot_stability() scatters per trial --
            # shows overall shape/spread/overlap between labels rather than
            # one point per trial_id.
            'roll_pitch_safe_limit_deg': 15.0,
            'bins': 20,
            'figsize': [11, 5],
            'legend_fontsize': 8,
            'dpi': 150,
            'title': 'Distribution of peak roll/pitch across the sweep',
            'colors': {},
        },
        'stability_timeseries_all': {
            # The "consider all the trials, not a couple of examples"
            # complement to plot_timeseries_examples() (which deliberately
            # shows only a handful of representative force/displacement
            # traces -- plotting every trial's full force signal would be
            # unreadable clutter; roll/pitch is compact enough that
            # literally every trial fits on one axis at once). Every
            # trial's roll/pitch trace is drawn as a thin line (line_alpha)
            # plus a per-label bold median line resampled onto a shared
            # n_grid_points-point time axis.
            'roll_pitch_safe_limit_deg': 15.0,
            'line_alpha': 0.15,
            'n_grid_points': 100,
            'figsize': [9, 7],
            'legend_fontsize': 7,
            'dpi': 150,
            'title': 'Roll/pitch over time since CONTACT_PROBE entry -- every trial (thin) + per-label median (bold)',
            'colors': {},
        },
        'safety_margin_backup': {
            'backup_min_clearance_m': _live_backup_min_clearance(),
            'figsize': [7, 5],
            'marker_size': 50,
            'violation_marker_size': 180,
            'legend_fontsize': 8,
            'point_color': 'tab:red',
            'threshold_color': 'black',
            'violation_color': 'red',
            'dpi': 150,
            'title': 'Backup safety margin vs configured minimum',
        },
        'estimator_health': {
            'figsize': [8, 6],
            'marker_size': 40,
            'legend_fontsize': 7,
            'dpi': 150,
            'colors': {},
        },
        'correlations': {
            'figsize': [11, 9],
            'marker_size': 40,
            'legend_fontsize': 6,
            'subplot_title_fontsize': 10,
            'dpi': 150,
            'colors': {},
        },
    }


# ── Data loading ─────────────────────────────────────────────────────────────

def load_summary(tag_dir, config):
    df = pd.read_csv(os.path.join(tag_dir, 'csv', 'summary.csv'))
    n_before = len(df)
    df = df[df['end_reason'] != 'superseded']   # forced-close artifacts, not real trials
    # A trial interrupted by a stack restart (run_sweep.sh) is re-run on
    # resume and can be logged twice; the later row is the complete one.
    df = df.drop_duplicates('trial_id', keep='last')
    # sim_anomaly (SimHealthMonitor, trial_core.py): a Gazebo physics
    # explosion/teleport, tip-over, out-of-range height, or a stall was
    # detected during the trial -- its data (timing, classification, even
    # peak forces) can't be trusted. Older summary.csv files predate this
    # column; skip the filter rather than KeyError on them.
    if 'sim_anomaly' in df.columns:
        df = df[df['sim_anomaly'].fillna('') == '']
    n_dropped = n_before - len(df)
    if n_dropped:
        print('[plotting] Excluded %d/%d rows (superseded/sim_anomaly) from %s' %
              (n_dropped, n_before, tag_dir))

    ocfg = config.get('outlier_filter', {})
    if ocfg.get('enabled', True):
        n_before_outliers = len(df)
        df, dropped = exclude_duration_outliers(
            df, ocfg.get('group_by', ['classified_label']),
            ocfg.get('value_col', 'nav_time_to_classification'),
            multiplier=ocfg.get('multiplier', 3.0), min_group_size=ocfg.get('min_group_size', 3))
        if len(dropped):
            value_col = ocfg.get('value_col', 'nav_time_to_classification')
            print('[plotting] Excluded %d/%d rows as duration outliers from %s:' %
                  (len(dropped), n_before_outliers, tag_dir))
            for _, row in dropped.iterrows():
                print('    trial_id=%s classified_label=%s %s=%.2f' %
                     (row['trial_id'], row['classified_label'], value_col, row[value_col]))

    gt = config['ground_truth']
    # ODE combines two contacting surfaces' friction via the minimum -- see
    # ground_truth.ground_mu's comment in default_config().
    effective_mu = np.minimum(df['mu'], gt.get('ground_mu', 1.0))
    df['static_force_n'] = effective_mu * df['mass_kg'] * G
    df['peak_force_horizontal'] = np.sqrt(df['peak_fx']**2 + df['peak_fy']**2)
    df['gt_movable_force_threshold'] = df['static_force_n'] < gt['classification_threshold_n']
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
    classification_threshold_n = gt_cfg['classification_threshold_n']
    push_capability_n = gt_cfg['push_capability_mass_fraction'] * gt_cfg['robot_mass_kg'] * G
    colors = dict(LABEL_COLORS, **cfg.get('colors', {}))

    geoms = sorted(df['geometry'].dropna().unique()) or [None]
    per_geom_w, h = cfg.get('figsize_per_geom', (6, 5))
    fig, axes = plt.subplots(1, len(geoms), figsize=(per_geom_w * len(geoms), h), squeeze=False)
    mass_lo = max(df['mass_kg'].min(skipna=True) * 0.9, 0.1)
    mass_hi = df['mass_kg'].max(skipna=True) * 1.05
    mass_curve = np.linspace(mass_lo, mass_hi, 200)

    for ax, geom in zip(axes[0], geoms):
        sub = df if geom is None else df[df['geometry'] == geom]
        for label, group in sub.groupby('classified_label'):
            ax.scatter(group['mass_kg'], group['mu'], c=colors.get(label, 'tab:brown'), label=label,
                      s=cfg.get('marker_size', 70), edgecolors='k', linewidths=0.5, zorder=3)

        if cfg.get('show_gt_boundaries', True):
            ax.plot(mass_curve, classification_threshold_n / (mass_curve * G), c='tab:purple', ls='--', lw=1.5,
                   label='GT boundary: classification threshold (%.0f N)' % classification_threshold_n, zorder=2)
            ax.plot(mass_curve, push_capability_n / (mass_curve * G), c='tab:olive', ls=':', lw=1.5,
                   label='GT boundary: %.0f%% robot weight (%.0f N)'
                         % (gt_cfg['push_capability_mass_fraction'] * 100, push_capability_n), zorder=2)

        ax.set_xlabel('mass (kg)')
        ax.set_ylabel('friction coefficient (mu)')
        # mu = force / (mass * g) blows up as mass -> 0 (e.g. a 128N boundary
        # at 1kg needs mu=13, nonsensical for a real material pair) -- both
        # curves are still drawn over the full mass range, they just exit the
        # top of the plot for low mass, which is the correct picture (that
        # boundary is simply outside any friction coefficient this sweep
        # tests). Capping the y-axis to the tested data's own range (instead
        # of letting matplotlib autoscale to the curve's low-mass blowup) is
        # what keeps the actually-relevant mu in [0, ~1] readable at all.
        mu_hi_data = df['mu'].max(skipna=True) if df['mu'].notna().any() else 1.0
        mu_ylim = cfg.get('mu_axis_max') or mu_hi_data * cfg.get('mu_axis_margin', 1.3)
        ax.set_ylim(0, mu_ylim)
        ax.set_title(geom or 'all geometries')
        ax.legend(fontsize=cfg.get('legend_fontsize', 7))
        ax.grid(alpha=0.3, zorder=0)
    fig.suptitle(cfg.get('title', 'Classification outcome vs mass / friction, with ground-truth boundaries'))
    fig.tight_layout(rect=(0, 0, 1, 0.92))  # reserve headroom so suptitle doesn't collide
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))   # with a subplot title when there's only 1 geometry
    plt.close(fig)


def _settled_classify_force(tag_dir, trial_ids, settle_time):
    """Per-trial mean sqrt(fx^2+fy^2) over the CLASSIFYING segment, from
    settle_time seconds after CLASSIFYING entry to its end -- an OFFLINE
    reconstruction of the "settled force" contact_probe_manager.py itself
    uses for force-mode classification (see its _wrench_forward_mag /
    classification_force_settle_time), computed here from the raw fx/fy
    columns already in the timeseries CSV rather than a live topic, so it
    works on any past results directory too.

    Uses horizontal magnitude sqrt(fx^2+fy^2) rather than the exact
    heading-projected component the live node uses (that needs
    probe_entry_yaw, which isn't logged per row) -- identical to the live
    value for a head-on approach (approach_angle_deg=0, the common case),
    a slight overestimate otherwise if there's a lateral component.
    Returns {trial_id: settled_force_n}, only for trials that reached
    CLASSIFYING at all.
    """
    out = {}
    for tid in trial_ids:
        path = os.path.join(tag_dir, 'csv', 'trial_%05d_timeseries.csv' % int(tid))
        if not os.path.exists(path):
            continue
        try:
            ts = pd.read_csv(path, usecols=['t_wall', 'fx', 'fy', 'state'])
        except Exception:
            continue
        cls = ts[ts['state'] == 'CLASSIFYING']
        if cls.empty:
            continue
        t0 = cls['t_wall'].iloc[0]
        settled = cls[cls['t_wall'] - t0 >= settle_time]
        if settled.empty:
            settled = cls   # settle_time >= classify window length edge case
        out[int(tid)] = float(np.sqrt(settled['fx']**2 + settled['fy']**2).mean())
    return out


def plot_force_vs_static_friction(df, out_path, cfg, tag_dir, settle_time):
    """Sanity check on the force ESTIMATOR (not the classification decision):
    does the settled contact force -- the same quantity force-mode
    classification actually decides on, reconstructed offline from each
    trial's CLASSIFYING segment -- scale sensibly with the physically-
    expected kinetic/static friction force (mu * mass * g)?

    Deliberately NOT the trial-wide peak_fx/peak_fy (what this plot used
    before): the peak includes impact transients and gait noise from the
    whole trial and is almost always well above any steady-state force, so
    a y=x comparison against it is close to meaningless -- everything sits
    above the line with no visible relationship. The settled force is what
    should actually track mu*mass*g under Coulomb friction (see
    contact_probe_manager.py's classification_mode='force' docs), so a
    linear fit + R^2 over the MOVABLE trials (the ones physically expected
    to reach steady sliding) is a real test of that relationship, not just
    an eyeballed reference line."""
    settled = _settled_classify_force(tag_dir, df['trial_id'], settle_time)
    df = df.copy()
    df['settled_force_n'] = df['trial_id'].map(settled)

    fig, ax = plt.subplots(figsize=cfg.get('figsize', (7, 5)))
    colors = dict(LABEL_COLORS, **cfg.get('colors', {}))
    for label, group in df.groupby('classified_label'):
        g = group.dropna(subset=['settled_force_n'])
        if g.empty:
            continue
        ax.scatter(g['static_force_n'], g['settled_force_n'],
                  c=colors.get(label, 'tab:brown'), label=label,
                  s=cfg.get('marker_size', 60), edgecolors='k', linewidths=0.5, zorder=3)

    movable = df[(df['classified_label'] == 'MOVABLE')].dropna(subset=['static_force_n', 'settled_force_n'])
    hi = max(df['static_force_n'].max(skipna=True),
             df['settled_force_n'].max(skipna=True), 1.0) * 1.05
    ax.plot([0, hi], [0, hi], ls='--', c='gray', lw=1, label='y = x (ideal Coulomb prediction)', zorder=2)
    if len(movable) >= 2 and movable['static_force_n'].std() > 0:
        slope, intercept = np.polyfit(movable['static_force_n'], movable['settled_force_n'], 1)
        pred = slope * movable['static_force_n'] + intercept
        ss_res = ((movable['settled_force_n'] - pred) ** 2).sum()
        ss_tot = ((movable['settled_force_n'] - movable['settled_force_n'].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        xs = np.array([0, hi])
        ax.plot(xs, slope * xs + intercept, ls='-', c='tab:green', lw=1.5, zorder=2,
               label='fit over MOVABLE: y=%.2fx%+.1f (R^2=%.2f)' % (slope, intercept, r2))

    ax.set_xlabel('theoretical friction force  min(mu, ground_mu) * mass * g  [N]')
    ax.set_ylabel('settled measured force  mean sqrt(fx^2+fy^2) post-settle  [N]')
    ax.set_title(cfg.get('title', 'Settled contact force vs expected friction'))
    ax.legend(fontsize=cfg.get('legend_fontsize', 8))
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
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
    convention). See gt_movable_force_threshold / gt_movable_mass_fraction in
    load_summary() for what gt_col means."""
    tp, fn, fp, tn, n_excluded = _confusion_counts(df, gt_col)
    mat = np.array([[tp, fn], [fp, tn]])
    total = tp + fn + fp + tn

    fig, ax = plt.subplots(figsize=cfg.get('figsize', (5.5, 5)))
    im = ax.imshow(mat, cmap=cfg.get('cmap', 'Blues'), vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha='center', va='center', fontsize=cfg.get('cell_fontsize', 18),
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
    ax.set_title('%s\n%s' % (gt_description, stats), fontsize=cfg.get('title_fontsize', 10))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
    plt.close(fig)


def _boxplot_by_label(ax, df, value_col, title, colors=None, title_fontsize=None):
    colors = dict(LABEL_COLORS, **(colors or {}))
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
        patch.set_facecolor(colors.get(label, 'tab:brown'))
        patch.set_alpha(0.5)
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(alpha=0.3, axis='y')
    return True


def plot_timing_by_label(df, out_path, cfg):
    """Three panels, deliberately separated because they answer different
    questions and conflating them is misleading:

    Left   -- time to DECISION (t_classify_entry -> t_classify_exit, i.e.
              classify_time). This is a fixed-duration window applied
              identically regardless of outcome, so it should be about the
              SAME for MOVABLE and SOLID -- there is no real "movable is
              faster to detect" effect; this panel exists to demonstrate that.
    Middle -- time PROBE-to-DECISION (t_probe_to_classify_exit_s:
              t_probe_entry -> t_classify_exit). Unlike the left panel this
              one is NOT a fixed window -- it also includes the
              approach-to-contact travel and the contact_hold_time dwell,
              both of which CAN vary with mass/mu (e.g. a high-friction/heavy
              object may ramp the wrench above threshold sooner than a
              light/slippery one that initially slips before load builds).
              Same field name as timing_trial_logger.py's (the
              lateral-length sweep) so classification speed is comparable
              across both sweeps -- see that module's docstring.
    Right  -- time to full RESOLUTION (nav_time_to_classification: goal-sent
              to state machine back in NORMAL_NAV). This one IS asymmetric,
              but for a structural reason: a SOLID verdict additionally
              requires the whole BACKING_UP maneuver before the state
              machine resolves, which a MOVABLE verdict skips entirely. The
              asymmetry lives here, not in the classification decision.
    """
    df = df.copy()
    df['time_to_decision_s'] = df['t_classify_exit'] - df['t_classify_entry']
    colors = cfg.get('colors', {})

    fig, axes = plt.subplots(1, 3, figsize=cfg.get('figsize', (15, 5)))
    ok1 = _boxplot_by_label(axes[0], df, 'time_to_decision_s',
                            'Time to classification DECISION\n(classify_time window -- same for both, by design)',
                            colors=colors)
    axes[0].set_ylabel('time (s)')
    ok2 = _boxplot_by_label(axes[1], df, 't_probe_to_classify_exit_s',
                            'Time PROBE-entry to DECISION\n(approach-to-contact + hold_time + classify_time -- can vary with mass/mu)',
                            colors=colors)
    axes[1].set_ylabel('time (s)')
    ok3 = _boxplot_by_label(axes[2], df, 'nav_time_to_classification',
                            'Time to full state-machine RESOLUTION\n(SOLID includes BACKING_UP; MOVABLE does not)',
                            colors=colors)
    axes[2].set_ylabel('time (s)')
    if not (ok1 or ok2 or ok3):
        raise ValueError('no timing data to plot')
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
    plt.close(fig)


def plot_diagnostics(df, out_path, cfg):
    """Node-health sanity check over the sweep: did contact_probe_manager's
    loop rate / wrench latency stay stable trial to trial?"""
    target_hz = cfg.get('loop_rate_target_hz', 10.0)
    ms = cfg.get('marker_size', 3)
    d = df.sort_values('trial_id')
    fig, axes = plt.subplots(2, 1, figsize=cfg.get('figsize', (8, 6)), sharex=True)
    axes[0].plot(d['trial_id'], d['cpm_loop_rate_hz_mean'], marker='o', ms=ms,
                c=cfg.get('loop_rate_color', 'tab:blue'))
    axes[0].axhline(target_hz, ls='--', c='gray', lw=1, label='target (%.0f Hz)' % target_hz)
    axes[0].set_ylabel('loop rate (Hz)')
    axes[0].legend(fontsize=cfg.get('legend_fontsize', 8))
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
    axes[1].plot(d['trial_id'], d['cpm_wrench_latency_ms_mean'], marker='o', ms=ms,
                c=cfg.get('latency_color', 'tab:orange'))
    axes[1].set_ylabel('wrench latency (ms)')
    axes[1].set_xlabel('trial_id')
    axes[1].grid(alpha=0.3)
    fig.suptitle(cfg.get('title', 'contact_probe_manager runtime diagnostics over the sweep'))
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
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

    state_colors = dict(STATE_SHADE_COLORS, **cfg.get('state_colors', {}))
    fmag_color = cfg.get('force_color', 'crimson')
    disp_color = cfg.get('displacement_color', 'tab:blue')
    row_height = cfg.get('row_height', 2.8)
    rolling_window = cfg.get('rolling_window', 50)

    fig, axes = plt.subplots(len(trial_ids), 1,
                             figsize=cfg.get('figsize', (9, row_height * len(trial_ids) + 0.6)),
                             squeeze=False)
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

        for state, color in state_colors.items():
            mask = (ts['state'] == state).values
            if mask.any():
                ax.fill_between(ts['t_since_probe_entry'], 0, 1, where=mask,
                               transform=ax.get_xaxis_transform(), color=color,
                               alpha=0.15, step='post', zorder=0)
                if state not in seen_states:
                    seen_states[state] = Patch(facecolor=color, alpha=0.3,
                                              label=STATE_SHADE_DESCRIPTIONS.get(state, state))

        # Raw |F| is dominated by gait-cycle noise (see CLAUDE.md's notes on
        # MomentumObserver walking-dynamics noise) -- plot it faint as
        # context, and a rolling-mean overlay as the readable signal. This
        # window is illustrative only, NOT a reproduction of
        # contact_probe_manager's own per-axis wrench_filter_window
        # averaging (it averages fx/fy/fz/tx/ty/tz separately against
        # separate thresholds, not this combined magnitude).
        fmag = np.sqrt(ts['fx']**2 + ts['fy']**2 + ts['fz']**2)
        fmag_smooth = fmag.rolling(rolling_window, min_periods=1, center=True).mean()
        ax.plot(ts['t_since_probe_entry'], fmag, c='k', lw=0.5, alpha=0.3, zorder=2)
        l1, = ax.plot(ts['t_since_probe_entry'], fmag_smooth, c=fmag_color, lw=1.5,
                      label='|F| (N), smoothed', zorder=3)
        ax2 = ax.twinx()
        l2, = ax2.plot(ts['t_since_probe_entry'], ts['forward_disp_since_probe'],
                       c=disp_color, lw=1, ls='--', label='forward disp (m)', zorder=3)
        if not line_handles:
            line_handles, line_labels = [l1, l2], [l1.get_label(), l2.get_label()]

        ax.set_title('trial %d -- %s (mass=%.1fkg, mu=%.2f)'
                    % (tid, row['classified_label'], row['mass_kg'], row['mu']))
        ax.set_ylabel('|F| (N)')
        ax2.set_ylabel('fwd disp (m)', color=disp_color)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('t since CONTACT_PROBE entry (s)')
    # One shared legend for the whole figure -- the state-color meaning and
    # line styles are identical across every row, so repeating a 6-entry
    # legend per subplot would just be clutter.
    handles = line_handles + [seen_states[s] for s in state_colors if s in seen_states]
    labels = line_labels + [h.get_label() for h in handles[len(line_handles):]]
    if handles:
        fig.legend(handles, labels, fontsize=cfg.get('legend_fontsize', 8), loc='upper center',
                  ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93 if len(trial_ids) <= 2 else 0.96))
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
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

    colors = dict(LABEL_COLORS, **cfg.get('colors', {}))
    marker_size = cfg.get('marker_size', 40)
    legend_fontsize = cfg.get('legend_fontsize', 6)
    d = df.sort_values('trial_id')
    fig, axes = plt.subplots(2, 2, figsize=cfg.get('figsize', (11, 8)))

    for ax, col, ylabel, safe_line in (
            (axes[0, 0], 'peak_roll_deg', 'peak |roll| (deg)', safe_deg),
            (axes[0, 1], 'peak_pitch_deg', 'peak |pitch| (deg)', safe_deg),
            (axes[1, 0], 'peak_roll_rate_dps', 'peak |roll rate| (deg/s)', None),
            (axes[1, 1], 'peak_pitch_rate_dps', 'peak |pitch rate| (deg/s)', None)):
        for label, group in d.groupby('classified_label'):
            ax.scatter(group['trial_id'], group[col], c=colors.get(label, 'tab:brown'), label=label,
                      s=marker_size, edgecolors='k', linewidths=0.4, zorder=3)
        if safe_line is not None:
            ax.axhline(safe_line, ls='--', c='red', lw=1, alpha=0.6,
                      label='reference limit (%.0f deg)' % safe_line, zorder=2)
        ax.set_xlabel('trial_id')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=legend_fontsize)
    fig.suptitle(cfg.get('title', 'Base stability (roll/pitch) across the sweep'))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
    plt.close(fig)


def _phase_stability_stats(tag_dir, trial_ids):
    """Per-trial std of roll/pitch/roll_rate/pitch_rate, split into
    'baseline' (state==NORMAL_NAV -- ordinary walking, before any probing
    starts) and 'contact' (state in CONTACT_PROBE/CLASSIFYING/BACKING_UP --
    probing, pushing, or reversing). One row per trial per phase, so every
    trial contributes equally regardless of how many timeseries rows it has.

    Reads trial_NNNNN_timeseries.csv directly from <tag_dir>/csv/ rather
    than trusting summary.csv's timeseries_csv_path column, which can be an
    absolute container path not valid on whatever machine this actually
    runs on -- tag_dir-relative reconstruction is portable.
    """
    cols = ['state', 'roll', 'pitch', 'roll_rate_dps', 'pitch_rate_dps']
    baseline_rows, contact_rows = [], []
    for tid in trial_ids:
        path = os.path.join(tag_dir, 'csv', 'trial_%05d_timeseries.csv' % int(tid))
        if not os.path.exists(path):
            continue
        try:
            ts = pd.read_csv(path, usecols=lambda c: c in cols)
        except Exception:
            continue
        if ts.empty or 'state' not in ts.columns:
            continue

        def _stats(sub):
            return {'trial_id': tid,
                   'roll_std_deg': np.degrees(sub['roll']).std(),
                   'pitch_std_deg': np.degrees(sub['pitch']).std(),
                   'roll_rate_std_dps': sub['roll_rate_dps'].std(),
                   'pitch_rate_std_dps': sub['pitch_rate_dps'].std()}

        base = ts[ts['state'] == 'NORMAL_NAV']
        if len(base) >= 5:   # a handful of samples minimum for a meaningful std
            baseline_rows.append(_stats(base))
        contact = ts[ts['state'].isin(['CONTACT_PROBE', 'CLASSIFYING', 'BACKING_UP'])]
        if len(contact) >= 5:
            contact_rows.append(_stats(contact))
    return pd.DataFrame(baseline_rows), pd.DataFrame(contact_rows)


def plot_stability_baseline(df, out_path, cfg, tag_dir):
    """Are the roll/pitch swings during probing/pushing/backing-up actually
    unusual, or just normal walking dynamics? plot_stability()'s peak values
    (and this instrumentation's own peak_roll_deg etc. in summary.csv) are
    trial-wide peaks, which include the ordinary walking approach phase too
    -- they can't answer this by themselves. This plot splits each trial's
    own timeseries into a NORMAL_NAV baseline segment and a
    CONTACT_PROBE/CLASSIFYING/BACKING_UP segment, computes the std of
    roll/pitch/roll-rate/pitch-rate within each, and compares the two
    distributions across the whole sweep -- baseline is the reference for
    "how much does this robot normally wobble while walking", so if contact
    isn't visibly worse than baseline, the roll/pitch data doesn't support a
    safety concern; if it's clearly larger, that's a real, sweep-wide effect
    rather than a couple of noisy trial-wide peaks."""
    baseline, contact = _phase_stability_stats(tag_dir, df['trial_id'])
    if baseline.empty:
        raise ValueError(
            'no NORMAL_NAV baseline segments found in any trial timeseries -- trial_logger.py '
            'only started recording from TrialStart (instead of CONTACT_PROBE entry) as of the '
            'fix that added this plot; results recorded before that node restart have no '
            'pre-contact rows at all, so there is nothing to use as a baseline. Restart '
            'trial_logger.py and re-run the sweep to get comparable data.')
    if contact.empty:
        raise ValueError('no CONTACT_PROBE/CLASSIFYING/BACKING_UP segments found -- nothing to '
                         'compare the baseline against (e.g. every trial was NO_CONTACT).')

    fig, axes = plt.subplots(2, 2, figsize=cfg.get('figsize', (11, 8)))
    panels = (
        (axes[0, 0], 'roll_std_deg', 'std(roll) per trial (deg)'),
        (axes[0, 1], 'pitch_std_deg', 'std(pitch) per trial (deg)'),
        (axes[1, 0], 'roll_rate_std_dps', 'std(roll rate) per trial (deg/s)'),
        (axes[1, 1], 'pitch_rate_std_dps', 'std(pitch rate) per trial (deg/s)'),
    )
    box_colors = cfg.get('colors', ['tab:gray', 'tab:orange'])
    for ax, col, ylabel in panels:
        data, labels = [], []
        if not baseline.empty and col in baseline:
            data.append(baseline[col].dropna().values)
            labels.append('baseline\n(NORMAL_NAV)\nn=%d' % len(baseline))
        if not contact.empty and col in contact:
            data.append(contact[col].dropna().values)
            labels.append('contact\n(PROBE/CLASSIFY/BACKUP)\nn=%d' % len(contact))
        if not data:
            ax.set_title(ylabel + '\n(no data)')
            continue
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle(cfg.get('title', 'Stability during contact vs normal-walking baseline\n'
                        '(per-trial std within each phase, pooled across the sweep)'))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
    plt.close(fig)


def plot_stability_vs_param(df, out_path, cfg):
    """Peak roll/pitch vs. the sweep's own independent variables (mass, mu)
    instead of plot_stability()'s trial_id (run order, which has no
    physical meaning). Answers the actually interesting question directly:
    does a heavier/higher-friction push correlate with more instability?"""
    needed = ['peak_roll_deg', 'peak_pitch_deg']
    _require_columns(df, needed, 'trial_logger.py needs restarting after the roll/pitch '
                                 'instrumentation was added (see its module docstring)')
    if df[needed].isna().all().all():
        raise ValueError('stability columns are present but entirely NaN for every trial in '
                         'this results directory -- recorded before trial_logger.py was restarted')

    colors = dict(LABEL_COLORS, **cfg.get('colors', {}))
    marker_size = cfg.get('marker_size', 40)
    legend_fontsize = cfg.get('legend_fontsize', 7)
    safe_deg = cfg.get('roll_pitch_safe_limit_deg', 15.0)

    fig, axes = plt.subplots(2, 2, figsize=cfg.get('figsize', (11, 8)))
    panels = (
        (axes[0, 0], 'mass_kg', 'peak_roll_deg', 'mass (kg)', 'peak |roll| (deg)'),
        (axes[0, 1], 'mu', 'peak_roll_deg', 'friction coefficient (mu)', 'peak |roll| (deg)'),
        (axes[1, 0], 'mass_kg', 'peak_pitch_deg', 'mass (kg)', 'peak |pitch| (deg)'),
        (axes[1, 1], 'mu', 'peak_pitch_deg', 'friction coefficient (mu)', 'peak |pitch| (deg)'),
    )
    any_plotted = False
    for ax, xcol, ycol, xlabel, ylabel in panels:
        sub = df.dropna(subset=[xcol, ycol])
        if sub.empty:
            ax.set_title(ylabel + ' vs ' + xlabel + '\n(no data)')
            ax.axis('off')
            continue
        any_plotted = True
        for label, group in sub.groupby('classified_label'):
            ax.scatter(group[xcol], group[ycol], c=colors.get(label, 'tab:brown'), label=label,
                      s=marker_size, edgecolors='k', linewidths=0.4, zorder=3)
        ax.axhline(safe_deg, ls='--', c='red', lw=1, alpha=0.6,
                  label='reference limit (%.0f deg)' % safe_deg, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=legend_fontsize)
    if not any_plotted:
        plt.close(fig)
        raise ValueError('no rows with both mass_kg/mu and peak roll/pitch data')
    fig.suptitle(cfg.get('title', "Base stability vs. the sweep's own independent variables "
                        '(mass, friction)'))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
    plt.close(fig)


def plot_stability_distribution(df, out_path, cfg):
    """Distribution (histogram) of peak roll/pitch across the WHOLE sweep,
    split by classified_label -- shows the overall shape/spread/overlap
    between labels' instability, complementing plot_stability()'s one
    point per trial and plot_stability_vs_param()'s per-parameter view."""
    needed = ['peak_roll_deg', 'peak_pitch_deg']
    _require_columns(df, needed, 'trial_logger.py needs restarting after the roll/pitch '
                                 'instrumentation was added (see its module docstring)')
    if df[needed].isna().all().all():
        raise ValueError('stability columns are present but entirely NaN for every trial in '
                         'this results directory -- recorded before trial_logger.py was restarted')

    colors = dict(LABEL_COLORS, **cfg.get('colors', {}))
    safe_deg = cfg.get('roll_pitch_safe_limit_deg', 15.0)
    bins = cfg.get('bins', 20)
    present = [l for l in df['classified_label'].dropna().unique()]
    ordered = [l for l in LABEL_COLORS if l in present] + [l for l in present if l not in LABEL_COLORS]

    fig, axes = plt.subplots(1, 2, figsize=cfg.get('figsize', (11, 5)))
    any_plotted = False
    for ax, col, xlabel in ((axes[0], 'peak_roll_deg', 'peak |roll| (deg)'),
                            (axes[1], 'peak_pitch_deg', 'peak |pitch| (deg)')):
        for label in ordered:
            vals = df.loc[df['classified_label'] == label, col].dropna().values
            if len(vals) == 0:
                continue
            any_plotted = True
            ax.hist(vals, bins=bins, color=colors.get(label, 'tab:brown'), alpha=0.5,
                   label='%s (n=%d)' % (label, len(vals)), edgecolor='k', linewidth=0.3)
        ax.axvline(safe_deg, ls='--', c='red', lw=1, alpha=0.6,
                  label='reference limit (%.0f deg)' % safe_deg)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('trial count')
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=cfg.get('legend_fontsize', 8))
    if not any_plotted:
        plt.close(fig)
        raise ValueError('no peak roll/pitch data to plot')
    fig.suptitle(cfg.get('title', 'Distribution of peak roll/pitch across the sweep'))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
    plt.close(fig)


def _collect_stability_traces(tag_dir, df):
    """Per-trial (classified_label, t_since_probe_entry, roll_deg,
    pitch_deg) arrays for every trial that has a timeseries file -- the
    same raw per-trial traces plot_timeseries_examples() reads for a
    couple of hand-picked examples, but for literally every trial, feeding
    plot_stability_timeseries_all()'s all-trials view instead."""
    out = []
    for tid, label in zip(df['trial_id'], df['classified_label']):
        path = os.path.join(tag_dir, 'csv', 'trial_%05d_timeseries.csv' % int(tid))
        if not os.path.exists(path):
            continue
        try:
            ts = pd.read_csv(path, usecols=lambda c: c in ('t_since_probe_entry', 'roll', 'pitch'))
        except Exception:
            continue
        if ts.empty or 't_since_probe_entry' not in ts.columns:
            continue
        out.append((label, ts['t_since_probe_entry'].values,
                   np.degrees(ts['roll'].values), np.degrees(ts['pitch'].values)))
    return out


def plot_stability_timeseries_all(df, out_path, cfg, tag_dir):
    """Roll/pitch over time since CONTACT_PROBE entry, for EVERY trial in
    the sweep at once -- the "consider all the trials, not a couple of
    examples" complement to plot_timeseries_examples() (which deliberately
    shows only a handful of representative force/displacement traces,
    since plotting every trial's full force signal would be unreadable
    clutter; roll/pitch is compact enough that literally all of them fit on
    one axis). Each trial is drawn as a thin, semi-transparent line; a bold
    per-label median line is overlaid, computed by resampling every trial
    onto a shared time grid (np.interp, NaN past a trial's own end) and
    taking nanmedian at each grid point."""
    traces = _collect_stability_traces(tag_dir, df)
    if not traces:
        raise ValueError('no per-trial timeseries files found (or none with data)')

    colors = dict(LABEL_COLORS, **cfg.get('colors', {}))
    line_alpha = cfg.get('line_alpha', 0.15)
    n_grid = cfg.get('n_grid_points', 100)
    safe_deg = cfg.get('roll_pitch_safe_limit_deg', 15.0)

    max_t = max((t.max() for _, t, _, _ in traces if len(t)), default=None)
    if max_t is None or max_t <= 0:
        raise ValueError('every trial timeseries is empty or single-sample -- nothing to plot')
    grid = np.linspace(0, max_t, n_grid)

    fig, axes = plt.subplots(2, 1, figsize=cfg.get('figsize', (9, 7)), sharex=True)
    by_label = {}
    for label, t, roll_deg, pitch_deg in traces:
        color = colors.get(label, 'tab:brown')
        axes[0].plot(t, roll_deg, color=color, alpha=line_alpha, lw=0.8, zorder=1)
        axes[1].plot(t, pitch_deg, color=color, alpha=line_alpha, lw=0.8, zorder=1)
        by_label.setdefault(label, []).append((t, roll_deg, pitch_deg))

    for label, rows in by_label.items():
        color = colors.get(label, 'tab:brown')
        roll_grid = np.full((len(rows), n_grid), np.nan)
        pitch_grid = np.full((len(rows), n_grid), np.nan)
        for i, (t, roll_deg, pitch_deg) in enumerate(rows):
            if len(t) < 2:
                continue
            roll_grid[i] = np.interp(grid, t, roll_deg, right=np.nan)
            pitch_grid[i] = np.interp(grid, t, pitch_deg, right=np.nan)
        # All-NaN columns are expected and correct here (e.g. MOVABLE trials
        # end sooner than SOLID's backup maneuver, so nothing has data past
        # that point for that label) -- nanmedian's RuntimeWarning about it
        # is just noise, not a real problem.
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore', category=RuntimeWarning)
            roll_med = np.nanmedian(roll_grid, axis=0)
            pitch_med = np.nanmedian(pitch_grid, axis=0)
        axes[0].plot(grid, roll_med, color=color, lw=2.5, zorder=3,
                    label='%s (median, n=%d)' % (label, len(rows)))
        axes[1].plot(grid, pitch_med, color=color, lw=2.5, zorder=3,
                    label='%s (median, n=%d)' % (label, len(rows)))

    for ax, ylabel in ((axes[0], 'roll (deg)'), (axes[1], 'pitch (deg)')):
        ax.axhline(safe_deg, ls='--', c='red', lw=1, alpha=0.5,
                  label='reference limit (±%.0f deg)' % safe_deg)
        ax.axhline(-safe_deg, ls='--', c='red', lw=1, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=cfg.get('legend_fontsize', 7))
    axes[-1].set_xlabel('t since CONTACT_PROBE entry (s)')
    fig.suptitle(cfg.get('title', 'Roll/pitch over time since CONTACT_PROBE entry -- every '
                        'trial (thin) + per-label median (bold)'))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
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

    fig, ax = plt.subplots(figsize=cfg.get('figsize', (7, 5)))
    ax.scatter(sub['trial_id'], sub['min_rear_clearance_during_backup_m'],
              c=cfg.get('point_color', 'tab:red'), s=cfg.get('marker_size', 50),
              edgecolors='k', linewidths=0.5, zorder=3, label='min clearance actually kept')
    ax.axhline(configured_min, ls='--', c=cfg.get('threshold_color', 'black'), lw=1.5,
              label='configured backup_min_clearance (%.2f m)' % configured_min, zorder=2)
    violated = sub[sub['min_rear_clearance_during_backup_m'] < configured_min]
    if not violated.empty:
        ax.scatter(violated['trial_id'], violated['min_rear_clearance_during_backup_m'],
                  facecolors='none', edgecolors=cfg.get('violation_color', 'red'),
                  s=cfg.get('violation_marker_size', 180), linewidths=2, zorder=4,
                  label='below configured minimum (%d trials)' % len(violated))
    ax.set_xlabel('trial_id')
    ax.set_ylabel('min clearance to unknown rear obstacle during BACKING_UP (m)')
    ax.set_title(cfg.get('title', 'Backup safety margin vs configured minimum'))
    ax.legend(fontsize=cfg.get('legend_fontsize', 8))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
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

    colors = dict(LABEL_COLORS, **cfg.get('colors', {}))
    marker_size = cfg.get('marker_size', 40)
    d = df.sort_values('trial_id')
    fig, axes = plt.subplots(2, 1, figsize=cfg.get('figsize', (8, 6)), sharex=True)
    for ax, col, ylabel in (
            (axes[0], 'peak_residual_err_int', 'peak |internal residual error|'),
            (axes[1], 'peak_residual_err_ext', 'peak |external residual error|')):
        for label, group in d.groupby('classified_label'):
            ax.scatter(group['trial_id'], group[col], c=colors.get(label, 'tab:brown'), label=label,
                      s=marker_size, edgecolors='k', linewidths=0.4, zorder=3)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=cfg.get('legend_fontsize', 7))
    axes[-1].set_xlabel('trial_id')
    fig.suptitle(cfg.get('title', 'HaptiQuad estimator residual error across the sweep\n'
                        "(higher = less trustworthy wrench reading at that moment)"))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
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
    colors = dict(LABEL_COLORS, **cfg.get('colors', {}))
    marker_size = cfg.get('marker_size', 40)
    fig, axes = plt.subplots(2, 2, figsize=cfg.get('figsize', (11, 9)))
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
            ax.scatter(group[xcol], group[ycol], c=colors.get(label, 'tab:brown'), label=label,
                      s=marker_size, edgecolors='k', linewidths=0.4, zorder=3)
        if len(sub) >= 2 and sub[xcol].std() > 0:
            m, b = np.polyfit(sub[xcol], sub[ycol], 1)
            xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 50)
            ax.plot(xs, m * xs + b, c='gray', ls='--', lw=1, alpha=0.7, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=cfg.get('subplot_title_fontsize', 10))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=cfg.get('legend_fontsize', 6))
    if not any_ok:
        raise ValueError('no correlation data available (missing columns / no rows)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get('dpi', 150))
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


_PLOT_CONFIG_FILE = os.path.join(os.path.dirname(_THIS_FILE), 'plot_config.py')


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
    config = load_or_init_config(tag_dir, default_config())
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
        config['force_vs_static_friction'], tag_dir, config['ground_truth']['classification_settle_time_s'])
    _run('confusion_matrix_force_threshold.png', plot_confusion_matrix,
        config['confusion_matrix_force_threshold'], 'gt_movable_force_threshold',
        'Ground truth: settled-force classification threshold (%.0f N)'
        % config['ground_truth']['classification_threshold_n'])
    _run('confusion_matrix_mass_fraction.png', plot_confusion_matrix,
        config['confusion_matrix_mass_fraction'], 'gt_movable_mass_fraction',
        'Ground truth: %.0f%% of robot weight (%.1f kg robot)'
        % (config['ground_truth']['push_capability_mass_fraction'] * 100,
           config['ground_truth']['robot_mass_kg']))
    _run('timing_by_label.png', plot_timing_by_label, config['timing_by_label'])
    _run('diagnostics_over_sweep.png', plot_diagnostics, config['diagnostics'])
    _run('timeseries_examples.png', plot_timeseries_examples, config['timeseries_examples'])
    _run('stability.png', plot_stability, config['stability'])
    _run('stability_baseline.png', plot_stability_baseline, config['stability_baseline'], tag_dir)
    _run('stability_vs_mass_mu.png', plot_stability_vs_param, config['stability_vs_param'])
    _run('stability_distribution.png', plot_stability_distribution, config['stability_distribution'])
    _run('stability_timeseries_all.png', plot_stability_timeseries_all,
        config['stability_timeseries_all'], tag_dir)
    _run('safety_margin_backup.png', plot_safety_margin_backup, config['safety_margin_backup'])
    _run('estimator_health.png', plot_estimator_health, config['estimator_health'])
    _run('correlations.png', plot_correlations, config['correlations'])

    bundle_scripts(tag_dir, [_THIS_FILE, _PLOT_CONFIG_FILE], _REGENERATE_SCRIPT)

    return written, errors
