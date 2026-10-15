#!/usr/bin/env python3
"""
plot_tuning_results.py — generate all plots for a completed (or
in-progress) tuning_sweep_orchestrator.py results directory. Pure
post-processing, no ROS runtime needed -- run this any time against any
past results directory (tuning_sweep_orchestrator.py also calls this
automatically when a sweep finishes, via
haptiquad_nav_eval.tuning_plotting.generate_all()).

Independent from plot_results.py (mass/mu sweep) and
plot_timing_results.py (lateral-length sweep) -- different results schema,
see tuning_plotting.py.

Usage:
    rosrun haptiquad_nav_eval plot_tuning_results.py <tag_dir>
    rosrun haptiquad_nav_eval plot_tuning_results.py <results_dir> <tag>
"""
import argparse
import os
import sys

from haptiquad_nav_eval.tuning_plotting import generate_all


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results_dir', help='Either a tag directory (contains a csv/ '
                        'subfolder with summary.csv) or a results root (pass `tag` too).')
    parser.add_argument('tag', nargs='?', default=None,
                        help='If given, results_dir/tag is used as the tag directory.')
    args = parser.parse_args()

    tag_dir = os.path.join(args.results_dir, args.tag) if args.tag else args.results_dir
    summary_path = os.path.join(tag_dir, 'csv', 'summary.csv')
    if not os.path.exists(summary_path):
        print('No summary.csv found at %s' % summary_path, file=sys.stderr)
        sys.exit(1)

    written, errors = generate_all(tag_dir)

    for path in written:
        print('wrote', path)
    for filename, e in errors:
        print('SKIPPED %s: %s' % (filename, e), file=sys.stderr)

    if not written:
        sys.exit(1)


if __name__ == '__main__':
    main()
