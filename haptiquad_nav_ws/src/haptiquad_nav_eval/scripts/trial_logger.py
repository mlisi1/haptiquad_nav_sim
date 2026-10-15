#!/usr/bin/env python3
"""
trial_logger.py — records extensive per-trial data for HaptiQuad-Nav
classification/navigation experiments.

Independent from sweep_orchestrator.py: it only listens. Ground truth
(mass, mu, geometry, approach params) comes entirely from the
orchestrator's TrialStart message -- this node never talks to Gazebo or
the world SDF. t_contact_onset is derived from the SAME contact_hold_time
param already loaded onto the param server by contact_probe_manager (read,
not duplicated as a hardcoded constant), so it stays correct if that param
is ever retuned.

The classification label (MOVABLE / SOLID / NO_CONTACT) is derived purely
from the /haptiquad_nav/state sequence, not from any dedicated "result"
message or independent ground-truth check: contact_probe_manager's own
threshold logic IS the thing under test here, so its own state sequence is
trusted as the result -- CLASSIFYING -> BACKING_UP implies SOLID,
CLASSIFYING -> NORMAL_NAV directly implies MOVABLE, and
CONTACT_PROBE -> NORMAL_NAV directly (probe timeout, no sustained contact)
implies NO_CONTACT.

nav_time_to_classification / nav_status_at_finish are NOT a navigation
success/failure outcome -- they cover t_goal_sent through the moment the
trial's classification resolves (state returns to NORMAL_NAV), which is
BEFORE contact_probe_manager's resent goal has any chance to actually
reach SUCCEEDED/ABORTED (the trial ends, and the object gets deleted,
right as that goal is resent). nav_status_at_finish is therefore almost
always a transient PREEMPTING/ACTIVE, not a real terminal outcome --
that's expected, not a bug. If you need genuine "did the robot actually
reach the goal" timing, the trial boundary needs to move to wait for a
real terminal /move_base/status instead, which adds real per-trial
runtime (the robot has to keep walking after classification resolves) --
worth a deliberate follow-up, not implied by these field names.

Safety/stability data (added on top of the original classification-focused
fields): peak roll/pitch and roll/pitch RATE (finite-differenced from
/slam/pose, no new topic needed), peak base-height deviation from the pose
at CONTACT_PROBE entry, minimum clearance to an unknown rear obstacle
specifically DURING BACKING_UP (from contact_probe_manager's new
/haptiquad_nav/nearest_(rear_)obstacle_distance topics -- compare against
contact_probe.yaml's backup_min_clearance to see how much margin was
actually used, not just configured), and peak HaptiQuad estimator residual
error (/residual_errors -- how much to trust the wrench signal at all
during the trial). None of this proves real-hardware safety; it is what
this simulation directly demonstrates. Requires contact_probe_manager and
this node to be RESTARTED after being updated for the new topics/fields to
populate -- older rows/trials recorded before that restart will have NaN
in these columns, which is expected, not a bug.

Per trial, writes (CSVs live under <tag_dir>/csv/, not the tag root, so the
tag directory's top level stays just csv/, plots/, plot_scripts/, and the
contact_probe_manager config snapshot -- see sweep_orchestrator.py):
  - <results_dir>/<tag>/csv/trial_<id>_timeseries.csv — one row per
    /anymal/base_wrench message received from TrialStart (so it includes the
    NORMAL_NAV/SLOW_APPROACH walk before any probing, not just
    CONTACT_PROBE onward -- needed as a baseline reference, see
    plot_stability_baseline in plotting.py) through the next NORMAL_NAV
    entry (native rate, zero-order-hold on the slower topics: pose, state,
    speed_limit, diagnostics).
  - <results_dir>/<tag>/csv/summary.csv — one appended row per trial
    (pandas-ready; header written once).

A per-trial watchdog timer guarantees exactly one summary row is written
per TrialStart message even if the state machine never reaches
CONTACT_PROBE at all (e.g. a missed approach) -- a trial is never silently
dropped, which matters for running unattended across a full sweep.
"""
import csv
import math
import os

import rospkg
import rospy
import tf.transformations

from std_msgs.msg import String, Float64
from geometry_msgs.msg import WrenchStamped, PoseWithCovarianceStamped
from diagnostic_msgs.msg import DiagnosticArray
from actionlib_msgs.msg import GoalStatusArray, GoalStatus
from haptiquad_msgs.msg import ResidualErrorStamped
from haptiquad_nav_eval.msg import TrialStart
from haptiquad_nav_eval.trial_core import SimHealthMonitor


def _default_results_dir():
    # Hidden folder inside the package itself (not the container's home
    # dir) so it lands under the host bind-mount (./haptiquad_nav_ws) and
    # is visible outside Docker. See package .gitignore -- .results/ is
    # deliberately excluded from git.
    return os.path.join(rospkg.RosPack().get_path('haptiquad_nav_eval'), '.results')


TIME_SERIES_FIELDS = [
    'trial_id', 't_wall', 't_since_probe_entry',
    'fx', 'fy', 'fz', 'tx', 'ty', 'tz',
    'pose_x', 'pose_y', 'pose_z',
    'roll', 'pitch', 'yaw', 'roll_rate_dps', 'pitch_rate_dps',
    'forward_disp_since_probe',
    'state', 'speed_limit',
    'nearest_obs_m', 'nearest_rear_obs_m',
    'residual_err_int_max', 'residual_err_ext_max',
    'cpm_loop_rate_hz', 'cpm_wrench_latency_ms',
]

SUMMARY_FIELDS = [
    'trial_id', 'experiment_tag', 'timestamp_start', 'object_name',
    'mass_kg', 'mu', 'geometry', 'size_x', 'size_y', 'size_z',
    'approach_angle_rad', 'approach_offset_m', 'spawn_x', 'spawn_y',
    'robot_origin_x', 'robot_origin_y', 'robot_origin_yaw',
    't_probe_entry', 't_contact_onset', 't_classify_entry', 't_classify_exit',
    # Mirrors timing_trial_logger.py's field of the same name (mass/mu sweep
    # vs. lateral-length sweep otherwise never share a schema -- see that
    # module's docstring) so "how long does classification actually take"
    # can be compared/plotted across both sweeps on equal terms: probe entry
    # (physical approach begins) through classify exit (decision made),
    # i.e. approach-to-contact + contact_hold_time + classify_time.
    't_probe_to_classify_exit_s',
    'delta_d_classify_window',
    'peak_fx', 'peak_fy', 'peak_fz', 'peak_tx', 'peak_ty', 'peak_tz',
    'classified_label', 'classification_mode', 'end_reason',
    # Set by SimHealthMonitor (trial_core.py) if a Gazebo physics
    # explosion/teleport, tip-over, out-of-range height, or a stall (no
    # progress while contact_probe_manager expected forward motion) was
    # detected at any point during the trial -- empty string if clean.
    # generate_all()/load_summary() in plotting.py excludes non-empty rows
    # from aggregate stats so one corrupted trial can't skew a cell's
    # mean/std. See tuning_sweep_orchestrator.py's SimHealthMonitor.anomaly()
    # for why this doesn't just try to prevent the underlying Gazebo glitch.
    'sim_anomaly',
    't_goal_sent', 'nav_time_to_classification',
    'nav_status_at_finish', 'nav_status_at_finish_text',
    # Stability / safety -- see module docstring's "Safety/stability data" note.
    'peak_roll_deg', 'peak_pitch_deg',
    'peak_roll_rate_dps', 'peak_pitch_rate_dps',
    'peak_base_height_delta_m',
    'min_rear_clearance_during_backup_m',
    'peak_residual_err_int', 'peak_residual_err_ext',
    'cpm_loop_rate_hz_mean', 'cpm_loop_rate_hz_min', 'cpm_loop_rate_hz_max',
    'cpm_wrench_latency_ms_mean', 'cpm_wrench_latency_ms_max',
    'n_timeseries_rows', 'timeseries_csv_path',
]

NAN = float('nan')

# actionlib_msgs/GoalStatus code -> name, for the human-readable summary column
_STATUS_NAMES = {
    GoalStatus.PENDING: 'PENDING', GoalStatus.ACTIVE: 'ACTIVE',
    GoalStatus.PREEMPTED: 'PREEMPTED', GoalStatus.SUCCEEDED: 'SUCCEEDED',
    GoalStatus.ABORTED: 'ABORTED', GoalStatus.REJECTED: 'REJECTED',
    GoalStatus.PREEMPTING: 'PREEMPTING', GoalStatus.RECALLING: 'RECALLING',
    GoalStatus.RECALLED: 'RECALLED', GoalStatus.LOST: 'LOST',
}


class TrialLogger:

    def __init__(self):
        self._results_dir = rospy.get_param('results_dir', _default_results_dir())
        # contact_hold_time/classification_mode live in contact_probe.yaml, which
        # is loaded into contact_probe_manager's PRIVATE namespace by navigation.launch
        # (a separate launch file from this node's own eval_sweep.launch) -- so they
        # must be read via contact_probe_manager's absolute param path, not a bare
        # name (a bare name here would resolve to /contact_hold_time, which nothing
        # ever sets, and silently fall back to the Python default below).
        self._contact_hold_time = rospy.get_param(
            '/contact_probe_manager/contact_hold_time', 1.0)
        # Ablation toggle -- recorded per row so a results folder produced with
        # classification_mode=force can't be silently mistaken for displacement
        # mode (or vice versa) during offline analysis.
        self._classification_mode = rospy.get_param(
            '/contact_probe_manager/classification_mode', 'force')
        # Watchdog must exceed the orchestrator's own trial_timeout so a clean
        # state-driven finish always wins the race under normal conditions.
        self._watchdog_s = rospy.get_param('trial_timeout', 60.0) + 15.0

        self._latest_pose = None          # (x, y, z, roll, pitch, yaw)
        self._prev_pose_sample = None     # (t, roll, pitch) for rate finite-differencing
        self._latest_roll_rate = NAN
        self._latest_pitch_rate = NAN
        self._latest_speed_limit = NAN
        self._latest_diag = {}            # loop_rate_hz, wrench_latency_ms
        self._latest_move_base_status = None   # (code, text)
        self._latest_nearest_obs = NAN
        self._latest_nearest_rear_obs = NAN
        self._latest_residual = {}        # err_int_max, err_ext_max (this message's peak |value|)
        self._state = None
        self._prev_state = None

        self._trial = None                # latest TrialStart
        self._recording = False
        self._finished_current = True     # no trial in flight yet
        self._watchdog_timer = None
        self._rows = []
        self._probe_entry_pos = None      # (x0, y0, yaw0)
        self._probe_entry_height = None   # pose_z at CONTACT_PROBE entry
        self._t_probe_entry = None
        self._t_classify_entry = None
        self._t_classify_exit = None
        self._d_at_classify_entry = None
        self._d_at_classify_exit = None
        self._peak = self._zero_peaks()
        self._peak_roll_deg = 0.0
        self._peak_pitch_deg = 0.0
        self._peak_roll_rate_dps = 0.0
        self._peak_pitch_rate_dps = 0.0
        self._peak_height_delta_m = 0.0
        self._min_rear_clearance_backup = None   # only tracked while state == BACKING_UP
        self._peak_residual_err_int = 0.0
        self._peak_residual_err_ext = 0.0
        self._loop_rates = []
        self._latencies = []
        self._health = SimHealthMonitor()

        rospy.Subscriber('/haptiquad_nav_eval/trial_start', TrialStart,
                         self._trial_start_cb, queue_size=1)
        rospy.Subscriber('/haptiquad_nav/state', String, self._state_cb, queue_size=10)
        rospy.Subscriber('/slam/pose', PoseWithCovarianceStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber('/haptiquad_nav/speed_limit', Float64, self._speed_cb, queue_size=1)
        rospy.Subscriber('/diagnostics', DiagnosticArray, self._diag_cb, queue_size=1)
        rospy.Subscriber('/anymal/base_wrench', WrenchStamped, self._wrench_cb, queue_size=50)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self._move_base_status_cb,
                         queue_size=5)
        rospy.Subscriber('/haptiquad_nav/nearest_obstacle_distance', Float64,
                         self._nearest_obs_cb, queue_size=1)
        rospy.Subscriber('/haptiquad_nav/nearest_rear_obstacle_distance', Float64,
                         self._nearest_rear_obs_cb, queue_size=1)
        rospy.Subscriber('/residual_errors', ResidualErrorStamped, self._residual_cb, queue_size=1)

        rospy.loginfo('[TrialLogger] Ready -- writing to %s', self._results_dir)

    @staticmethod
    def _zero_peaks():
        return {k: 0.0 for k in ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')}

    # ── cheap caches ──────────────────────────────────────────────────────────

    def _pose_cb(self, msg):
        p = msg.pose.pose
        q = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(q)
        self._latest_pose = (p.position.x, p.position.y, p.position.z, roll, pitch, yaw)

        t = msg.header.stamp.to_sec()
        if self._prev_pose_sample is not None:
            t0, roll0, pitch0 = self._prev_pose_sample
            dt = t - t0
            if dt > 1e-4:  # guard against duplicate/out-of-order stamps
                self._latest_roll_rate = math.degrees(roll - roll0) / dt
                self._latest_pitch_rate = math.degrees(pitch - pitch0) / dt
        self._prev_pose_sample = (t, roll, pitch)

        if not self._recording:
            return
        roll_deg, pitch_deg = math.degrees(roll), math.degrees(pitch)
        self._peak_roll_deg = max(self._peak_roll_deg, abs(roll_deg))
        self._peak_pitch_deg = max(self._peak_pitch_deg, abs(pitch_deg))
        if not math.isnan(self._latest_roll_rate):
            self._peak_roll_rate_dps = max(self._peak_roll_rate_dps, abs(self._latest_roll_rate))
        if not math.isnan(self._latest_pitch_rate):
            self._peak_pitch_rate_dps = max(self._peak_pitch_rate_dps, abs(self._latest_pitch_rate))
        if self._probe_entry_height is not None:
            self._peak_height_delta_m = max(
                self._peak_height_delta_m, abs(p.position.z - self._probe_entry_height))

    def _speed_cb(self, msg):
        self._latest_speed_limit = msg.data

    def _diag_cb(self, msg):
        for status in msg.status:
            if status.name != 'contact_probe_manager':
                continue
            kv = {v.key: v.value for v in status.values}
            try:
                self._latest_diag = {
                    'loop_rate_hz': float(kv.get('loop_rate_hz', 'nan')),
                    'wrench_latency_ms': float(kv.get('wrench_latency_ms', 'nan')),
                }
            except ValueError:
                pass

    def _nearest_obs_cb(self, msg):
        self._latest_nearest_obs = msg.data

    def _nearest_rear_obs_cb(self, msg):
        self._latest_nearest_rear_obs = msg.data
        # Safety-critical window only: how close did the robot actually get
        # to an unknown rear obstacle DURING the reversing maneuver, vs
        # contact_probe.yaml's configured backup_min_clearance. Outside
        # BACKING_UP this distance isn't a safety concern (nothing behind
        # the robot is being approached), so it's not tracked as a "peak".
        if self._recording and self._state == 'BACKING_UP' and math.isfinite(msg.data):
            self._min_rear_clearance_backup = (msg.data if self._min_rear_clearance_backup is None
                                               else min(self._min_rear_clearance_backup, msg.data))

    def _residual_cb(self, msg):
        err_int_max = max((abs(v) for v in msg.err_int), default=0.0)
        err_ext_max = max((abs(v) for v in msg.err_ext), default=0.0)
        self._latest_residual = {'err_int_max': err_int_max, 'err_ext_max': err_ext_max}
        if self._recording:
            self._peak_residual_err_int = max(self._peak_residual_err_int, err_int_max)
            self._peak_residual_err_ext = max(self._peak_residual_err_ext, err_ext_max)

    def _move_base_status_cb(self, msg):
        # Whatever goal is most recently reported -- deliberately not
        # edge-tracked. contact_probe_manager preempts/resends goals of its
        # own during a trial; we only care what move_base last reported by
        # the time the trial concludes (see _finish_trial).
        if msg.status_list:
            last = msg.status_list[-1]
            self._latest_move_base_status = (last.status, last.text)

    # ── trial lifecycle ───────────────────────────────────────────────────────

    def _trial_start_cb(self, msg):
        if not self._finished_current:
            rospy.logwarn(
                '[TrialLogger] Trial %d started before trial %d was finished -- '
                'forcing an early close on the previous one (check orchestrator pacing).',
                msg.trial_id, self._trial.trial_id if self._trial else -1)
            self._finish_trial(prev_state_at_exit=self._state, end_reason='superseded')

        self._trial = msg
        # Recording starts at TrialStart, not CONTACT_PROBE entry, so the
        # NORMAL_NAV (and SLOW_APPROACH) walk before any probing is captured
        # too -- this is the only way to get a genuine "normal gait" baseline
        # to compare contact-phase stability against (see plot_stability_baseline
        # in plotting.py). t_since_probe_entry is NaN for these pre-probe rows
        # since _t_probe_entry isn't set until CONTACT_PROBE is actually entered.
        self._recording = True
        self._finished_current = False
        self._rows = []
        self._probe_entry_pos = None
        self._probe_entry_height = None
        self._t_probe_entry = None
        self._t_classify_entry = None
        self._t_classify_exit = None
        self._d_at_classify_entry = None
        self._d_at_classify_exit = None
        self._peak = self._zero_peaks()
        self._peak_roll_deg = 0.0
        self._peak_pitch_deg = 0.0
        self._peak_roll_rate_dps = 0.0
        self._peak_pitch_rate_dps = 0.0
        self._peak_height_delta_m = 0.0
        self._min_rear_clearance_backup = None
        self._peak_residual_err_int = 0.0
        self._peak_residual_err_ext = 0.0
        self._loop_rates = []
        self._latencies = []
        self._latest_move_base_status = None
        self._health.arm()

        if self._watchdog_timer is not None:
            self._watchdog_timer.shutdown()
        self._watchdog_timer = rospy.Timer(
            rospy.Duration(self._watchdog_s), self._watchdog_cb, oneshot=True)

        rospy.loginfo('[TrialLogger] Trial %d started (%s, mass=%.1fkg mu=%.2f)',
                      msg.trial_id, msg.geometry, msg.mass, msg.mu)

    def _watchdog_cb(self, _event):
        if not self._finished_current:
            rospy.logwarn(
                '[TrialLogger] Trial %d: watchdog fired without a clean state-machine '
                'finish -- forcing a summary row (end_reason=watchdog_timeout).',
                self._trial.trial_id)
            self._finish_trial(prev_state_at_exit=self._state, end_reason='watchdog_timeout')

    def _state_cb(self, msg):
        self._prev_state = self._state
        self._state = msg.data

        if self._trial is None or self._finished_current:
            return

        # These four checks are deliberately independent `if`s, not an
        # elif chain: a direct CLASSIFYING -> NORMAL_NAV transition (the
        # MOVABLE case) must hit BOTH the "classify exit" bookkeeping AND
        # the "finish trial" check in the same callback, since msg.data
        # is 'NORMAL_NAV' and self._prev_state is 'CLASSIFYING'
        # simultaneously. An elif here silently swallows the finish check
        # -- the trial then only gets closed out (with a stale, wrong
        # prev_state) whenever the NEXT unrelated NORMAL_NAV message
        # happens to arrive, which is a real bug that shipped once and
        # mislabeled every direct-to-movable trial as INCOMPLETE.
        if msg.data == 'CONTACT_PROBE' and self._prev_state != 'CONTACT_PROBE':
            self._recording = True
            self._t_probe_entry = rospy.Time.now().to_sec()
            if self._latest_pose is not None:
                x, y, z, _, _, yaw = self._latest_pose
                self._probe_entry_pos = (x, y, yaw)
                self._probe_entry_height = z

        if msg.data == 'CLASSIFYING' and self._prev_state != 'CLASSIFYING':
            self._t_classify_entry = rospy.Time.now().to_sec()
            self._d_at_classify_entry = self._forward_disp()

        if self._prev_state == 'CLASSIFYING' and msg.data != 'CLASSIFYING':
            self._t_classify_exit = rospy.Time.now().to_sec()
            self._d_at_classify_exit = self._forward_disp()

        if msg.data == 'NORMAL_NAV' and self._recording:
            self._recording = False
            self._finish_trial(prev_state_at_exit=self._prev_state, end_reason='completed')

    def _forward_disp(self):
        if self._probe_entry_pos is None or self._latest_pose is None:
            return NAN
        x0, y0, yaw0 = self._probe_entry_pos
        x, y = self._latest_pose[0], self._latest_pose[1]
        return (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)

    # ── native-rate time series ──────────────────────────────────────────────

    def _wrench_cb(self, msg):
        if not self._recording or self._trial is None:
            return

        w = msg.wrench
        for k, v in (('fx', w.force.x), ('fy', w.force.y), ('fz', w.force.z),
                    ('tx', w.torque.x), ('ty', w.torque.y), ('tz', w.torque.z)):
            self._peak[k] = max(self._peak[k], abs(v))

        if 'loop_rate_hz' in self._latest_diag:
            self._loop_rates.append(self._latest_diag['loop_rate_hz'])
        if 'wrench_latency_ms' in self._latest_diag:
            self._latencies.append(self._latest_diag['wrench_latency_ms'])

        pose = self._latest_pose or (NAN, NAN, NAN, NAN, NAN, NAN)
        row = {
            'trial_id': self._trial.trial_id,
            't_wall': msg.header.stamp.to_sec(),
            't_since_probe_entry': (msg.header.stamp.to_sec() - self._t_probe_entry
                                    if self._t_probe_entry is not None else NAN),
            'fx': w.force.x, 'fy': w.force.y, 'fz': w.force.z,
            'tx': w.torque.x, 'ty': w.torque.y, 'tz': w.torque.z,
            'pose_x': pose[0], 'pose_y': pose[1], 'pose_z': pose[2],
            'roll': pose[3], 'pitch': pose[4], 'yaw': pose[5],
            'roll_rate_dps': self._latest_roll_rate, 'pitch_rate_dps': self._latest_pitch_rate,
            'forward_disp_since_probe': self._forward_disp(),
            'state': self._state,
            'speed_limit': self._latest_speed_limit,
            'nearest_obs_m': self._latest_nearest_obs,
            'nearest_rear_obs_m': self._latest_nearest_rear_obs,
            'residual_err_int_max': self._latest_residual.get('err_int_max', NAN),
            'residual_err_ext_max': self._latest_residual.get('err_ext_max', NAN),
            'cpm_loop_rate_hz': self._latest_diag.get('loop_rate_hz', NAN),
            'cpm_wrench_latency_ms': self._latest_diag.get('wrench_latency_ms', NAN),
        }
        self._rows.append(row)

    # ── trial finalization ───────────────────────────────────────────────────

    @staticmethod
    def _mean(xs):
        return sum(xs) / len(xs) if xs else NAN

    def _finish_trial(self, prev_state_at_exit, end_reason):
        if self._finished_current or self._trial is None:
            return
        self._finished_current = True
        self._recording = False
        if self._watchdog_timer is not None:
            self._watchdog_timer.shutdown()
            self._watchdog_timer = None

        trial = self._trial
        tag_dir = os.path.join(self._results_dir, trial.experiment_tag)
        csv_dir = os.path.join(tag_dir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)

        ts_path = os.path.join(csv_dir, 'trial_%05d_timeseries.csv' % trial.trial_id)
        with open(ts_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=TIME_SERIES_FIELDS)
            w.writeheader()
            w.writerows(self._rows)

        if prev_state_at_exit == 'BACKING_UP':
            classified_label = 'SOLID'
        elif prev_state_at_exit == 'CLASSIFYING':
            classified_label = 'MOVABLE'
        elif prev_state_at_exit == 'CONTACT_PROBE':
            classified_label = 'NO_CONTACT'
        else:
            classified_label = 'INCOMPLETE'   # watchdog fired before CONTACT_PROBE was ever seen

        t_contact_onset = (self._t_classify_entry - self._contact_hold_time
                           if self._t_classify_entry is not None else NAN)
        t_probe_to_classify_exit = (
            self._t_classify_exit - self._t_probe_entry
            if (self._t_probe_entry is not None and self._t_classify_exit is not None)
            else NAN)
        delta_d = (self._d_at_classify_exit - self._d_at_classify_entry
                  if (self._d_at_classify_entry is not None
                      and self._d_at_classify_exit is not None) else NAN)

        t_goal_sent = trial.header.stamp.to_sec()
        t_finish = rospy.Time.now().to_sec()
        # NOT a navigation outcome -- see the module docstring. This is
        # whatever move_base last reported by the time CLASSIFICATION
        # resolved, almost always a transient PREEMPTING/ACTIVE rather
        # than a real SUCCEEDED/ABORTED.
        nav_status_at_finish, nav_status_at_finish_text = (NAN, '')
        if self._latest_move_base_status is not None:
            code, text = self._latest_move_base_status
            nav_status_at_finish = _STATUS_NAMES.get(code, str(code))
            nav_status_at_finish_text = text

        row = {
            'trial_id': trial.trial_id,
            'experiment_tag': trial.experiment_tag,
            'timestamp_start': t_goal_sent,
            'object_name': trial.object_name,
            'mass_kg': trial.mass,
            'mu': trial.mu,
            'geometry': trial.geometry,
            'size_x': trial.size_x, 'size_y': trial.size_y, 'size_z': trial.size_z,
            'approach_angle_rad': trial.approach_angle,
            'approach_offset_m': trial.approach_offset,
            'spawn_x': trial.spawn_x, 'spawn_y': trial.spawn_y,
            'robot_origin_x': trial.robot_origin_x, 'robot_origin_y': trial.robot_origin_y,
            'robot_origin_yaw': trial.robot_origin_yaw,
            't_probe_entry': self._t_probe_entry if self._t_probe_entry is not None else NAN,
            't_contact_onset': t_contact_onset,
            't_classify_entry': self._t_classify_entry if self._t_classify_entry is not None else NAN,
            't_classify_exit': self._t_classify_exit if self._t_classify_exit is not None else NAN,
            't_probe_to_classify_exit_s': t_probe_to_classify_exit,
            'delta_d_classify_window': delta_d,
            'peak_fx': self._peak['fx'], 'peak_fy': self._peak['fy'], 'peak_fz': self._peak['fz'],
            'peak_tx': self._peak['tx'], 'peak_ty': self._peak['ty'], 'peak_tz': self._peak['tz'],
            'classified_label': classified_label,
            'classification_mode': self._classification_mode,
            'end_reason': end_reason,
            'sim_anomaly': self._health.anomaly() or '',
            't_goal_sent': t_goal_sent,
            'nav_time_to_classification': t_finish - t_goal_sent,
            'nav_status_at_finish': nav_status_at_finish,
            'nav_status_at_finish_text': nav_status_at_finish_text,
            'peak_roll_deg': self._peak_roll_deg,
            'peak_pitch_deg': self._peak_pitch_deg,
            'peak_roll_rate_dps': self._peak_roll_rate_dps,
            'peak_pitch_rate_dps': self._peak_pitch_rate_dps,
            'peak_base_height_delta_m': self._peak_height_delta_m,
            'min_rear_clearance_during_backup_m': (self._min_rear_clearance_backup
                                                   if self._min_rear_clearance_backup is not None
                                                   else NAN),
            'peak_residual_err_int': self._peak_residual_err_int,
            'peak_residual_err_ext': self._peak_residual_err_ext,
            'cpm_loop_rate_hz_mean': self._mean(self._loop_rates),
            'cpm_loop_rate_hz_min': min(self._loop_rates) if self._loop_rates else NAN,
            'cpm_loop_rate_hz_max': max(self._loop_rates) if self._loop_rates else NAN,
            'cpm_wrench_latency_ms_mean': self._mean(self._latencies),
            'cpm_wrench_latency_ms_max': max(self._latencies) if self._latencies else NAN,
            'n_timeseries_rows': len(self._rows),
            'timeseries_csv_path': ts_path,
        }

        summary_path = os.path.join(csv_dir, 'summary.csv')
        new_file = not os.path.exists(summary_path)
        with open(summary_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)

        rospy.loginfo('[TrialLogger] Trial %d finished: %s (%d rows, %.1fs to classification) -> %s',
                      trial.trial_id, classified_label, len(self._rows),
                      row['nav_time_to_classification'], ts_path)


if __name__ == '__main__':
    rospy.init_node('trial_logger')
    TrialLogger()
    rospy.spin()
