#!/usr/bin/env python3
"""
tuning_trial_logger.py — records one summary row per trial for the
HaptiQuad-Nav CONTACT-PROBE PARAMETER-TUNING harness
(tuning_sweep_orchestrator.py).

Independent from trial_logger.py (mass/mu sweep) and timing_trial_logger.py
(lateral-length sweep): separate message type (TuningTrialStart), separate
topic, separate results directory and CSV schema, own watchdog. All three
loggers can run against the same live navigation.launch stack without
interfering with each other, though only one sweep should be actively
producing trials at a time.

Uses ClassificationWatcher (trial_core.py) -- the same primitive
tuning_sweep_orchestrator.py uses to decide when to move to the next trial
-- to independently detect the exact moment CLASSIFYING resolves (to
NORMAL_NAV for MOVABLE, or to BACKING_UP for SOLID) and end the trial right
there. Deliberately does NOT wait for BACKING_UP to complete (unlike
trial_logger.py) or for the real goal to be reached (unlike
timing_trial_logger.py's GoalCompletionWatcher) -- this sweep only cares
about classification time.

Also tracks two things NOT strictly "classification time" but directly
relevant to judging whether a tuned value is actually SAFE to use, since
that's the tradeoff this whole sweep exists to characterize:
  - peak roll/pitch (finite-differenced from /slam/pose) -- watches for
    probe_speed (harder impact) or speed_factor (EXPERIMENTAL Gazebo
    speed-up) destabilizing the robot.
  - mean loop rate / wrench latency (from /diagnostics) -- directly
    relevant to wrench_filter_window, which trades gait-noise rejection
    against detection latency.

Per trial, writes <results_dir>/<tag>/csv/summary.csv (pandas-ready, one
appended row, header written once). A per-trial watchdog guarantees exactly
one summary row is written per TuningTrialStart even if classification is
never reached (e.g. no contact) -- a trial is never silently dropped.
"""
import csv
import math
import os

import rospkg
import rospy
import tf.transformations

from geometry_msgs.msg import PoseWithCovarianceStamped
from diagnostic_msgs.msg import DiagnosticArray
from haptiquad_nav_eval.msg import TuningTrialStart
from haptiquad_nav_eval.trial_core import ClassificationWatcher, SimHealthMonitor


def _default_results_dir():
    return os.path.join(rospkg.RosPack().get_path('haptiquad_nav_eval'), '.results')


SUMMARY_FIELDS = [
    'trial_id', 'experiment_tag', 'trial_mode', 'timestamp_start',
    'object_name', 'mass_kg', 'mu', 'geometry', 'size_x', 'size_y', 'size_z',
    'spawn_x', 'spawn_y', 'goal_x', 'goal_y',
    'approach_angle_rad', 'robot_origin_x', 'robot_origin_y', 'robot_origin_yaw',
    'combo_id',
    'probe_speed', 'probe_distance', 'contact_hold_time', 'classify_time',
    'classification_force_settle_time', 'wrench_filter_window',
    't_goal_sent', 't_probe_entry', 't_contact_onset',
    't_classify_entry', 't_classify_exit',
    't_probe_to_classify_exit_s', 'time_to_decision_s',
    'classified_label', 'end_reason', 'final_state',
    'peak_roll_deg', 'peak_pitch_deg',
    'cpm_loop_rate_hz_mean', 'cpm_wrench_latency_ms_mean',
    # Set by SimHealthMonitor (trial_core.py) if a Gazebo physics
    # explosion/teleport, tip-over, out-of-range height, or a stall was
    # detected during the trial -- empty string if clean. Complements
    # peak_roll_deg/peak_pitch_deg above with an independent position-jump
    # test, so a real launch (large jump, possibly without extreme
    # orientation) and a pure single-sample euler-angle artifact (extreme
    # roll/pitch with no accompanying jump) can be told apart instead of
    # both just showing up as "peak_roll_deg is huge" with no further
    # signal. Excluded from aggregate stats by tuning_plotting.py's
    # load_summary().
    'sim_anomaly',
]

NAN = float('nan')


class TuningTrialLogger:

    def __init__(self):
        self._results_dir = rospy.get_param('results_dir', _default_results_dir())
        self._watchdog_s = rospy.get_param('trial_timeout', 60.0) + 15.0

        self._watcher = ClassificationWatcher()
        self._health = SimHealthMonitor()

        self._trial = None
        self._recording = False
        self._finished_current = True
        self._watchdog_timer = None
        self._finish_check_timer = None

        self._prev_pose_sample = None   # (t, roll, pitch) for rate/peak tracking
        self._peak_roll_deg = 0.0
        self._peak_pitch_deg = 0.0
        self._loop_rates = []
        self._latencies = []

        rospy.Subscriber('/haptiquad_nav_eval/tuning_trial_start', TuningTrialStart,
                         self._trial_start_cb, queue_size=1)
        rospy.Subscriber('/slam/pose', PoseWithCovarianceStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber('/diagnostics', DiagnosticArray, self._diag_cb, queue_size=1)

        rospy.loginfo('[TuningTrialLogger] Ready -- writing to %s', self._results_dir)

    # ── native-rate tracking (only accumulated while a trial is in flight) ────

    def _pose_cb(self, msg):
        if not self._recording:
            return
        p = msg.pose.pose
        q = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        roll, pitch, _ = tf.transformations.euler_from_quaternion(q)
        roll_deg, pitch_deg = math.degrees(roll), math.degrees(pitch)
        self._peak_roll_deg = max(self._peak_roll_deg, abs(roll_deg))
        self._peak_pitch_deg = max(self._peak_pitch_deg, abs(pitch_deg))

    def _diag_cb(self, msg):
        if not self._recording:
            return
        for status in msg.status:
            if status.name != 'contact_probe_manager':
                continue
            kv = {v.key: v.value for v in status.values}
            try:
                self._loop_rates.append(float(kv.get('loop_rate_hz', 'nan')))
                self._latencies.append(float(kv.get('wrench_latency_ms', 'nan')))
            except ValueError:
                pass

    # ── trial lifecycle ───────────────────────────────────────────────────────

    def _trial_start_cb(self, msg):
        if not self._finished_current:
            rospy.logwarn(
                '[TuningTrialLogger] Trial %d started before trial %d was finished -- '
                'forcing an early close on the previous one (check orchestrator pacing).',
                msg.trial_id, self._trial.trial_id if self._trial else -1)
            self._finish_trial(end_reason='superseded')

        self._trial = msg
        self._recording = True
        self._finished_current = False
        self._watcher.arm()
        self._health.arm()
        self._peak_roll_deg = 0.0
        self._peak_pitch_deg = 0.0
        self._loop_rates = []
        self._latencies = []

        if self._watchdog_timer is not None:
            self._watchdog_timer.shutdown()
        self._watchdog_timer = rospy.Timer(
            rospy.Duration(self._watchdog_s), self._watchdog_cb, oneshot=True)

        if self._finish_check_timer is not None:
            self._finish_check_timer.shutdown()
        self._finish_check_timer = rospy.Timer(rospy.Duration(0.1), self._finish_check_cb)

        rospy.loginfo(
            '[TuningTrialLogger] Trial %d started (%s, combo %d, mass=%.1fkg mu=%.2f)',
            msg.trial_id, msg.trial_mode, msg.combo_id,
            msg.mass, msg.mu)

    def _watchdog_cb(self, _event):
        if not self._finished_current:
            rospy.logwarn(
                '[TuningTrialLogger] Trial %d: watchdog fired without a classification -- '
                'forcing a summary row (end_reason=watchdog_timeout).',
                self._trial.trial_id)
            self._finish_trial(end_reason='watchdog_timeout')

    def _finish_check_cb(self, _event):
        if self._finished_current or self._trial is None:
            return
        if self._watcher.resolved():
            self._finish_trial(end_reason='completed')

    @staticmethod
    def _seg(a, b):
        return (b - a) if (a is not None and b is not None) else NAN

    @staticmethod
    def _mean(xs):
        return sum(xs) / len(xs) if xs else NAN

    def _finish_trial(self, end_reason):
        if self._finished_current or self._trial is None:
            return
        self._finished_current = True
        self._recording = False
        if self._watchdog_timer is not None:
            self._watchdog_timer.shutdown()
            self._watchdog_timer = None
        if self._finish_check_timer is not None:
            self._finish_check_timer.shutdown()
            self._finish_check_timer = None

        trial = self._trial
        csv_dir = os.path.join(self._results_dir, trial.experiment_tag, 'csv')
        os.makedirs(csv_dir, exist_ok=True)

        exited_to = self._watcher.exited_to
        if exited_to == 'BACKING_UP':
            classified_label = 'SOLID'
        elif exited_to == 'NORMAL_NAV':
            classified_label = 'MOVABLE'
        elif self._watcher.saw_contact_probe:
            classified_label = 'NO_CONTACT'
        else:
            classified_label = 'INCOMPLETE'   # watchdog fired before CONTACT_PROBE was ever seen

        t_goal_sent = trial.header.stamp.to_sec()
        t_probe_entry = self._watcher.t_probe_entry
        t_classify_entry = self._watcher.t_classify_entry
        t_classify_exit = self._watcher.t_classify_exit
        # trial.contact_hold_time is THIS trial's actual value (it may be
        # the swept parameter itself), not contact_probe_manager's default --
        # using the logged per-trial value keeps this correct even when
        # contact_hold_time is what's being varied.
        t_contact_onset = (t_classify_entry - trial.contact_hold_time
                           if t_classify_entry is not None else NAN)

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
            'combo_id': trial.combo_id,
            'probe_speed': trial.probe_speed, 'probe_distance': trial.probe_distance,
            'contact_hold_time': trial.contact_hold_time, 'classify_time': trial.classify_time,
            'classification_force_settle_time': trial.classification_force_settle_time,
            'wrench_filter_window': trial.wrench_filter_window,
            't_goal_sent': t_goal_sent,
            't_probe_entry': t_probe_entry if t_probe_entry is not None else NAN,
            't_contact_onset': t_contact_onset,
            't_classify_entry': t_classify_entry if t_classify_entry is not None else NAN,
            't_classify_exit': t_classify_exit if t_classify_exit is not None else NAN,
            't_probe_to_classify_exit_s': self._seg(t_probe_entry, t_classify_exit),
            'time_to_decision_s': self._seg(t_classify_entry, t_classify_exit),
            'classified_label': classified_label,
            'end_reason': end_reason,
            'final_state': self._watcher.state,
            'peak_roll_deg': self._peak_roll_deg,
            'peak_pitch_deg': self._peak_pitch_deg,
            'cpm_loop_rate_hz_mean': self._mean(self._loop_rates),
            'cpm_wrench_latency_ms_mean': self._mean(self._latencies),
            'sim_anomaly': self._health.anomaly() or '',
        }

        summary_path = os.path.join(csv_dir, 'summary.csv')
        new_file = not os.path.exists(summary_path)
        with open(summary_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)

        rospy.loginfo(
            '[TuningTrialLogger] Trial %d finished: %s/%s (combo %d, %.2fs probe->exit) -> %s',
            trial.trial_id, trial.trial_mode, classified_label, trial.combo_id,
            row['t_probe_to_classify_exit_s'], summary_path)


if __name__ == '__main__':
    rospy.init_node('tuning_trial_logger')
    TuningTrialLogger()
    rospy.spin()
