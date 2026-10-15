"""
plot_config.py — shared per-plot YAML config loader, statistical duration-
outlier filter, and self-contained-bundle helper for plotting.py (mass/mu
sweep), timing_plotting.py (lateral-length sweep), and tuning_plotting.py
(contact-probe parameter sweep).

Each of those three modules defines its own default_config() -- a dict with
one section per plot filename stem, holding every tunable for that plot
(title, axis labels, legend text, colors, figure size, dpi, and whatever
else that plot's own docstring exposes). generate_all() in each module
calls load_or_init_config(tag_dir, default_config()) once, up front, to get
back a fully resolved config dict, then passes each plot function its own
cfg section. See any of the three modules' default_config() for the full
list of what's tunable for that sweep's plots.

<tag_dir>/plots/config/<name>.yaml is read back if already present (so
hand-edits survive a re-run -- edit the YAML, re-run the same command, only
that plot changes) or written from the defaults if not. Missing individual
files/keys fall back to defaults rather than erroring, so adding a new plot
(or a new key to an existing plot's defaults) later doesn't require
regenerating old results directories' config folders.

No ROS import here (mirrors the three callers' own "runs standalone, no
ROS required" constraint -- see e.g. plotting.py's module docstring), so
plot_scripts/regenerate.py (bundle_scripts, below) works from a bare
`pip install pandas numpy matplotlib pyyaml` even with no ROS present.
"""
import os
import shutil

try:
    import yaml
except ImportError:  # pragma: no cover -- normally present in this ROS env
    yaml = None


def load_or_init_config(tag_dir, defaults):
    """defaults: {section_name: {key: default_value, ...}, ...} -- one
    section per plot filename stem, from the caller's own default_config().

    Returns a same-shaped dict, each section resolved key-by-key: whatever
    is already on disk at <tag_dir>/plots/config/<name>.yaml overrides the
    matching default key, and every key the file doesn't mention keeps its
    current default -- so a hand-edited YAML that only touches one or two
    keys still gets every other key's up-to-date default, including ones
    added to default_config() after that YAML was first written."""
    config_dir = os.path.join(tag_dir, 'plots', 'config')
    os.makedirs(config_dir, exist_ok=True)
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
            with open(path, 'w', encoding='utf-8') as f:
                # allow_unicode: several defaults contain real arrows/×
                # (e.g. timing_plotting.py's segment labels) -- without it
                # PyYAML escapes them as \uXXXX, which is fine to load back
                # but unreadable for anyone hand-editing the file, exactly
                # the use case this config system exists for.
                yaml.safe_dump(section, f, sort_keys=False, allow_unicode=True)
    return resolved


def exclude_duration_outliers(df, group_by, value_col, multiplier=4.0, min_group_size=3):
    """Flags rows whose value_col is more than `multiplier` times that row's
    own group's MEDIAN value_col as a likely undetected sim glitch, not a
    real data point -- median rather than mean specifically because a single
    huge outlier barely moves a median but massively inflates a mean,
    which is exactly what makes these worth catching separately.

    This exists because SimHealthMonitor (trial_core.py) only catches
    genuinely STOPPED robots (zero net movement for a continuous window),
    a tip-over, or a position teleport -- confirmed, from a live run
    (test_timing_opt_speed), to miss cases where the robot keeps moving
    (however slowly/erratically) but the trial still takes 4-10x longer
    than every comparable trial for no logged reason (no state-machine
    re-triggers, no warnings -- see that investigation's findings). A
    "trial took 180s when the group's median is 20s" pattern is exactly
    what this catches that the online detector structurally cannot.

    Median is computed PER GROUP (group_by, e.g. ['trial_mode'] -- arm) so a
    genuinely slower-but-legitimate condition (e.g. a longer obstacle) isn't
    penalized relative to a faster one; groups with fewer than
    min_group_size non-null samples are left entirely unfiltered (too few
    points for a stable median -- better to keep a maybe-bad trial than to
    let a 2-sample group flag one of them as "the outlier" on a coin flip).
    Rows with a null value_col are never flagged (nothing to compare).

    Returns (kept_df, dropped_df) -- dropped_df is returned (not just a
    count) so a caller can log exactly which trial_ids/values were excluded:
    unlike sim_anomaly's deterministic reason string, this is a statistical
    judgment call the run itself never flagged, so it's worth being visible
    about specifically what got dropped and why."""
    if value_col not in df.columns or df.empty:
        return df, df.iloc[0:0]
    valid = df[value_col].notna()
    medians = df.groupby(group_by)[value_col].transform('median')
    counts = df.groupby(group_by)[value_col].transform('count')  # non-null count per group
    is_outlier = valid & (counts >= min_group_size) & (df[value_col] > multiplier * medians)
    return df[~is_outlier], df[is_outlier]


def bundle_scripts(tag_dir, module_files, regenerate_script):
    """Copy module_files (this results directory's plotting module PLUS
    this file, plot_config.py, since every caller imports it) into
    <tag_dir>/plot_scripts/, then write regenerate_script (a caller-supplied
    string -- each of the three plotting modules' entry point/import name
    differs) as plot_scripts/regenerate.py, so the results folder is
    self-contained: the plotting code, its shared config loader, and the
    config it was run with all travel together with the data.

    module_files: list of absolute paths to .py files to copy alongside.
    Copying the bundle's own already-in-place copy of itself (running
    regenerate.py FROM inside plot_scripts/, the documented primary use) is
    a silent no-op rather than an error -- shutil.copyfile refuses
    same-path copies (SameFileError) otherwise.
    """
    bundle_dir = os.path.join(tag_dir, 'plot_scripts')
    os.makedirs(bundle_dir, exist_ok=True)
    for src in module_files:
        dest = os.path.join(bundle_dir, os.path.basename(src))
        if os.path.abspath(dest) != os.path.abspath(src):
            shutil.copyfile(src, dest)
    with open(os.path.join(bundle_dir, 'regenerate.py'), 'w') as f:
        f.write(regenerate_script)
    os.chmod(os.path.join(bundle_dir, 'regenerate.py'), 0o755)
