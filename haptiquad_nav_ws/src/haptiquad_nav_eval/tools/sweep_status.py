#!/usr/bin/env python3
"""sweep_status.py -- progress and ETA of haptiquad_nav_eval sweeps.

    sweep_status.py <tag_dir> [<tag_dir> ...]          print status
    sweep_status.py --sample <tag_dir>                 (run_sweep.sh only)
        append "<epoch> <done>" to runner_logs/progress.log and rewrite
        <tag_dir>/STATUS

Plain Python 3, no ROS/pandas, so it runs on the host as well as in the
container. The rate is taken over the last RATE_WINDOW_S of samples (so it
includes stack restarts, which really do cost time) and falls back to the
whole run when that window saw no progress.
"""
import datetime
import os
import sys
import time

RATE_WINDOW_S = 3600.0


def _count_done(tag_dir):
    path = os.path.join(tag_dir, 'csv', 'progress.csv')
    try:
        with open(path) as f:
            return len({line.strip() for line in f.readlines()[1:] if line.strip()})
    except OSError:
        return 0


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _samples(tag_dir):
    path = os.path.join(tag_dir, 'runner_logs', 'progress.log')
    out = []
    try:
        with open(path) as f:
            for line in f:
                t, d = line.split()
                out.append((float(t), int(d)))
    except (OSError, ValueError):
        pass
    return out


def _rate(samples):
    """Seconds per trial, or None."""
    if len(samples) < 2:
        return None
    t_last, d_last = samples[-1]
    for window in (RATE_WINDOW_S, float('inf')):
        first = next(((t, d) for t, d in samples if t_last - t <= window), None)
        if first and d_last > first[1]:
            return (t_last - first[0]) / (d_last - first[1])
    return None


def _fmt_dur(s):
    s = int(s)
    return '%dh%02dm' % (s // 3600, (s % 3600) // 60)


def status(tag_dir):
    tag = os.path.basename(os.path.normpath(tag_dir))
    total = _read_int(os.path.join(tag_dir, 'TOTAL'))
    done = _count_done(tag_dir)
    if os.path.exists(os.path.join(tag_dir, 'DONE')):
        return '%s: FINISHED (%d trials)' % (tag, done)
    if total is None:
        return '%s: not started yet' % tag
    samples = _samples(tag_dir)
    attempts = 0
    logs = os.path.join(tag_dir, 'runner_logs')
    if os.path.isdir(logs):
        attempts = len([d for d in os.listdir(logs) if d.startswith('attempt_')])
    line = '%s: %d/%d trials (%.1f%%), %d remaining' % (
        tag, done, total, 100.0 * done / max(total, 1), total - done)
    rate = _rate(samples)
    if rate is not None:
        eta = (total - done) * rate
        finish = datetime.datetime.now() + datetime.timedelta(seconds=eta)
        line += ', %.1f s/trial, ETA %s (~%s)' % (rate, _fmt_dur(eta), finish.strftime('%a %H:%M'))
    else:
        line += ', ETA unknown (no rate yet)'
    if samples and time.time() - samples[-1][0] > 120:
        line += ' -- runner not sampling for %s (stopped?)' % _fmt_dur(time.time() - samples[-1][0])
    if attempts > 1:
        line += ' [stack attempt %d]' % attempts
    return line


def main(argv):
    if len(argv) == 3 and argv[1] == '--sample':
        tag_dir = argv[2]
        os.makedirs(os.path.join(tag_dir, 'runner_logs'), exist_ok=True)
        with open(os.path.join(tag_dir, 'runner_logs', 'progress.log'), 'a') as f:
            f.write('%d %d\n' % (time.time(), _count_done(tag_dir)))
        with open(os.path.join(tag_dir, 'STATUS'), 'w') as f:
            f.write(status(tag_dir) + '\n')
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 2
    for tag_dir in argv[1:]:
        print(status(tag_dir))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
