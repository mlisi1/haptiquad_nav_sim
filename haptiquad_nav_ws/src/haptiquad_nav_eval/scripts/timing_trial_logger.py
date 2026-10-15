#!/usr/bin/env python3
"""
timing_trial_logger.py — records one summary row per trial for the
HaptiQuad-Nav TIME-COMPARISON harness (timing_sweep_orchestrator.py).

Independent from trial_logger.py / TrialStart: separate message type
(TimingTrialStart), separate topic, separate results directory and CSV
schema, own watchdog. Both loggers can run against the same live
navigation.launch stack without interfering with each other, though only one
sweep should be actively producing trials at a time.

Unlike trial_logger.py (which ends a trial at classification resolution --
see that module's own docstring on why that's deliberate for the mass/mu
sweep), this logger's entire point is measuring REAL goal-reached time: a
trial finishes when GoalCompletionWatcher.reached() is true (ground-truth
position from /slam/pose within goal_tolerance_m of the commanded goal,
settled for goal_settle_time, with contact_probe_manager back in NORMAL_NAV)
or the watchdog fires -- see trial_core.py's GoalCompletionWatcher docstring
for why position-based detection is used instead of watching
/move_base/status directly (it races contact_probe_manager's own goal
resend after classification).

No native-rate time series is recorded here -- this experiment's output is
segment/total durations and the classification label, not fine-grained
stability data (trial_logger.py already covers that for the mass/mu sweep).

Per trial, writes <results_dir>/<tag>/csv/summary.csv (pandas-ready, one
appended row, header written once). A per-trial watchdog guarantees exactly
one summary row is written per TimingTrialStart even if the goal is never
reached (e.g. the robot gets stuck) -- a trial is never silently dropped.
"""
import csv
import os

import rospkg
import rospy

from std_msgs.msg import String
from haptiquad_nav_eval.msg import TimingTrialStart
from haptiquad_nav_eval.trial_core import GoalCompletionWatcher, SimHealthMonitor


def _default_results_dir():
    return os.path.join(rospkg.RosPack().get_path('haptiquad_nav_eval'), '.results')


SUMMARY_FIELDS = [
    'trial_id', 'experiment_tag', 'trial_mode', 'timestamp_start',
    'object_name', 'mass_kg', 'mu', 'geometry', 'size_x', 'size_y', 'size_z',
    'spawn_x', 'spawn_y', 'goal_x', 'goal_y',
    'approach_angle_rad', 'robot_origin_x', 'robot_origin_y', 'robot_origin_yaw',
    'robot_width_m',
    't_goal_sent', 't_probe_entry', 't_classify_entry', 't_classify_exit',
    't_backup_entry', 't_backup_exit', 't_finish',
    'total_time_s', 't_pre_contact_s', 't_probe_to_classify_exit_s',
    't_backup_s', 't_post_classify_to_finish_s',
    'classified_label', 'end_reason', 'final_state',
    # Set by SimHealthMonitor (trial_core.py) if a Gazebo physics
    # explosion/teleport, tip-over, out-of-range height, or a stall was
    # detected during the trial -- empty string if clean. Excluded from
    # aggregate stats by timing_plotting.py's load_summary().
    'sim_anomaly',
]

NAN = float('nan')


class TimingTrialLogger:

    def __init__(self):
        self._results_dir = rospy.get_param('results_dir', _default_results_dir())
        goal_tolerance_m = rospy.get_param('goal_tolerance_m', 0.30)
        goal_settle_time = rospy.get_param('goal_settle_time', 1.0)
        # Must exceed the orchestrator's own trial_timeout so a clean
        # position-driven finish always wins the race under normal conditions.
        self._watchdog_s = rospy.get_param('trial_timeout', 180.0) + 20.0

        self._watcher = GoalCompletionWatcher(goal_tolerance_m, goal_settle_time)
        self._health = SimHealthMonitor()

        self._trial = None
        self._finished_current = True   # no trial in flight yet
        self._watchdog_timer = None
        self._finish_check_timer = None

        self._state = None
        self._prev_state = None
        self._t_probe_entry = None
        self._t_classify_entry = None
        self._t_classify_exit = None
        self._t_backup_entry = None
        self._t_backup_exit = None

        rospy.Subscriber('/haptiquad_nav_eval/timing_trial_start', TimingTrialStart,
                         self._trial_start_cb, queue_size=1)
        rospy.Subscriber('/haptiquad_nav/state', String, self._state_cb, queue_size=10)

        rospy.loginfo('[TimingTrialLogger] Ready -- writing to %s', self._results_dir)

    # ── trial lifecycle ───────────────────────────────────────────────────────

    def _trial_start_cb(self, msg):
        if not self._finished_current:
            rospy.logwarn(
                '[TimingTrialLogger] Trial %d started before trial %d was finished -- '
                'forcing an early close on the previous one (check orchestrator pacing).',
                msg.trial_id, self._trial.trial_id if self._trial else -1)
            self._finish_trial(end_reason='superseded')

        self._trial = msg
        self._finished_current = False
        self._t_probe_entry = None
        self._t_classify_entry = None
        self._t_classify_exit = None
        self._t_backup_entry = None
        self._t_backup_exit = None
        # Deliberately NOT reset here: self._state/_prev_state carry over from
        # whatever the last /haptiquad_nav/state message said (contact_probe_manager
        # is a long-lived node -- a "classical" trial's bypass forces exactly
        # one fresh NORMAL_NAV message right before this callback fires, via
        # ~set_probing_enabled(false); see contact_probe_manager.py's
        # _set_probing_enabled_cb).

        self._watcher.arm(msg.goal_x, msg.goal_y)
        self._health.arm()

        if self._watchdog_timer is not None:
            self._watchdog_timer.shutdown()
        self._watchdog_timer = rospy.Timer(
            rospy.Duration(self._watchdog_s), self._watchdog_cb, oneshot=True)

        if self._finish_check_timer is not None:
            self._finish_check_timer.shutdown()
        self._finish_check_timer = rospy.Timer(rospy.Duration(0.1), self._finish_check_cb)

        rospy.loginfo(
            '[TimingTrialLogger] Trial %d started (%s, %s, mass=%.1fkg mu=%.2f size_y=%.2f)',
            msg.trial_id, msg.trial_mode, msg.geometry, msg.mass, msg.mu, msg.size_y)

    def _watchdog_cb(self, _event):
        if not self._finished_current:
            rospy.logwarn(
                '[TimingTrialLogger] Trial %d: watchdog fired without reaching the goal -- '
                'forcing a summary row (end_reason=watchdog_timeout).',
                self._trial.trial_id)
            self._finish_trial(end_reason='watchdog_timeout')

    def _finish_check_cb(self, _event):
        if self._finished_current or self._trial is None:
            return
        if self._watcher.reached():
            self._finish_trial(end_reason='completed')

    def _state_cb(self, msg):
        self._prev_state = self._state
        self._state = msg.data

        if self._trial is None or self._finished_current:
            return

        if msg.data == 'CONTACT_PROBE' and self._prev_state != 'CONTACT_PROBE':
            self._t_probe_entry = rospy.Time.now().to_sec()
        if msg.data == 'CLASSIFYING' and self._prev_state != 'CLASSIFYING':
            self._t_classify_entry = rospy.Time.now().to_sec()
        if self._prev_state == 'CLASSIFYING' and msg.data != 'CLASSIFYING':
            self._t_classify_exit = rospy.Time.now().to_sec()
        if msg.data == 'BACKING_UP' and self._prev_state != 'BACKING_UP':
            self._t_backup_entry = rospy.Time.now().to_sec()
        if self._prev_state == 'BACKING_UP' and msg.data != 'BACKING_UP':
            self._t_backup_exit = rospy.Time.now().to_sec()

    # ── trial finalization ───────────────────────────────────────────────────

    @staticmethod
    def _seg(a, b):
        return (b - a) if (a is not None and b is not None) else NAN

    def _finish_trial(self, end_reason):
        if self._finished_current or self._trial is None:
            return
        self._finished_current = True
        if self._watchdog_timer is not None:
            self._watchdog_timer.shutdown()
            self._watchdog_timer = None
        if self._finish_check_timer is not None:
            self._finish_check_timer.shutdown()
            self._finish_check_timer = None

        trial = self._trial
        csv_dir = os.path.join(self._results_dir, trial.experiment_tag, 'csv')
        os.makedirs(csv_dir, exist_ok=True)

        if trial.trial_mode == 'classical':
            classified_label = 'CLASSICAL'
        elif self._t_backup_entry is not None:
            classified_label = 'SOLID'
        elif self._t_classify_exit is not None:
            classified_label = 'MOVABLE'
        elif self._t_probe_entry is not None:
            classified_label = 'NO_CONTACT'
        else:
            classified_label = 'INCOMPLETE'

        t_goal_sent = trial.header.stamp.to_sec()
        t_finish = rospy.Time.now().to_sec()
        post_classify_start = (self._t_backup_exit if self._t_backup_exit is not None
                               else self._t_classify_exit)

        row = {
            'trial_id': trial.trial_id,
            'experiment_tag': trial.experiment_tag,
            'trial_mode': trial.trial_mode,
            'timestamp_start': t_goal_sent,
            'object_name': trial.object_name,
            'mass_kg': trial.mass, 'mu': trial.mu, 'geometry': trial.geometry,
            'size_x': trial.size_x, 'size_y': trial.size_y, 'size_z': trial.size_z,
            'spawn_x': trial.spawn_x, 'spawn_y': trial.spawn_y,
            'goal_x': trial.goal_x, 'goal_y': trial.goal_y,
            'approach_angle_rad': trial.approach_angle,
            'robot_origin_x': trial.robot_origin_x, 'robot_origin_y': trial.robot_origin_y,
            'robot_origin_yaw': trial.robot_origin_yaw,
            'robot_width_m': trial.robot_width_m,
            't_goal_sent': t_goal_sent,
            't_probe_entry': self._t_probe_entry if self._t_probe_entry is not None else NAN,
            't_classify_entry': self._t_classify_entry if self._t_classify_entry is not None else NAN,
            't_classify_exit': self._t_classify_exit if self._t_classify_exit is not None else NAN,
            't_backup_entry': self._t_backup_entry if self._t_backup_entry is not None else NAN,
            't_backup_exit': self._t_backup_exit if self._t_backup_exit is not None else NAN,
            't_finish': t_finish,
            'total_time_s': t_finish - t_goal_sent,
            't_pre_contact_s': self._seg(t_goal_sent, self._t_probe_entry),
            't_probe_to_classify_exit_s': self._seg(self._t_probe_entry, self._t_classify_exit),
            't_backup_s': self._seg(self._t_backup_entry, self._t_backup_exit),
            't_post_classify_to_finish_s': self._seg(post_classify_start, t_finish),
            'classified_label': classified_label,
            'end_reason': end_reason,
            'final_state': self._state,
            'sim_anomaly': self._health.anomaly() or '',
        }

        summary_path = os.path.join(csv_dir, 'summary.csv')
        new_file = not os.path.exists(summary_path)
        with open(summary_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)

        rospy.loginfo('[TimingTrialLogger] Trial %d finished: %s/%s (%.1fs total) -> %s',
                      trial.trial_id, trial.trial_mode, classified_label,
                      row['total_time_s'], summary_path)


if __name__ == '__main__':
    rospy.init_node('timing_trial_logger')
    TimingTrialLogger()
    rospy.spin()
