#!/usr/bin/env python3
"""
tuning_sweep_orchestrator.py — drives the contact-probe PARAMETER-TUNING
sweep: for each contact-probe timing parameter in tuning_sweep.yaml's
sweep_params, tries every value in its param_grids entry (holding every
OTHER tunable parameter at its baseline value), against two fixed arms --
movable and solid (mass/mu held fixed, matching timing_sweep.yaml's
representative points).

Completely independent from sweep_orchestrator.py and
timing_sweep_orchestrator.py: separate config (tuning_sweep.yaml), separate
message type (TuningTrialStart), separate results directory, separate
progress checkpointing. May run against the same long-lived
navigation.launch stack (contact_probe_manager, move_base) as the other two,
but never at the same time as either.

Per trial:
  1. Delete the previous object (if any) and fully reset the simulation
     (SimReset), same as the other two orchestrators.
  2. Explicitly set contact_probe_manager's probing bypass ON (there is no
     "classical" arm in this experiment -- probing always runs) and publish
     overrides for all six tunable parameters: baseline for every one
     except this trial's swept_param, which gets this trial's grid value.
     Always published explicitly every trial, never left over from a
     previous trial's override -- same discipline as
     timing_sweep_orchestrator.py's radius overrides.
  3. Spawn the (fixed-size) object, publish TuningTrialStart, send the
     approach goal.
  4. Wait for ClassificationWatcher.resolved() (CLASSIFYING has been
     exited, to either NORMAL_NAV or BACKING_UP) or trial_timeout, with a
     sanity warning if CONTACT_PROBE is never even entered. Deliberately
     does NOT wait for BACKING_UP to complete for the solid arm -- see
     tuning_sweep.yaml's "Trial end condition" comment.

Does NOT compute or store results itself -- tuning_trial_logger.py,
listening independently, is the single source of truth for logged data.
"""
import csv
import math
import os

import rospkg
import rospy
from std_msgs.msg import Float64
from std_srvs.srv import SetBool

from haptiquad_nav_eval.trial_core import (ObjectSpawner, SimReset, GoalSender, PoseCache,
                                           ClassificationWatcher, SimHealthMonitor,
                                           RealTimeFactorController, plan_object_and_goal)
from haptiquad_nav_eval.msg import TuningTrialStart

OBJECT_NAME_PREFIX = 'tuningobj_'
ARMS = ('movable', 'solid')

OVERRIDE_TOPICS = {
    'probe_speed': '/contact_probe_manager/probe_speed_override',
    'probe_distance': '/contact_probe_manager/probe_distance_override',
    'contact_hold_time': '/contact_probe_manager/contact_hold_time_override',
    'classify_time': '/contact_probe_manager/classify_time_override',
    'classification_force_settle_time':
        '/contact_probe_manager/classification_force_settle_time_override',
    'wrench_filter_window': '/contact_probe_manager/wrench_filter_window_override',
}


def _default_results_dir():
    return os.path.join(rospkg.RosPack().get_path('haptiquad_nav_eval'), '.results')


class TuningSweepOrchestrator:

    def __init__(self):
        self._tag = rospy.get_param('experiment_tag', 'tuning_sweep')
        self._geometry = rospy.get_param('geometry', 'box')
        self._size_x = rospy.get_param('size_x', 0.5)
        self._size_y = rospy.get_param('size_y', 0.5)
        self._size_z = rospy.get_param('size_z', 1.0)
        self._movable_mass = rospy.get_param('movable_mass_kg', 5.0)
        self._movable_mu = rospy.get_param('movable_mu', 0.1)
        self._solid_mass = rospy.get_param('solid_mass_kg', 200.0)
        self._solid_mu = rospy.get_param('solid_mu', 0.8)
        self._approach_distance = rospy.get_param('approach_distance', 3.0)
        self._goal_overshoot = rospy.get_param('goal_overshoot', 1.5)
        self._settle_time = rospy.get_param('settle_time', 1.0)
        self._reset_settle_time = rospy.get_param('reset_settle_time', 1.5)
        self._no_contact_timeout = rospy.get_param('no_contact_timeout', 25.0)
        self._trial_timeout = rospy.get_param('trial_timeout', 60.0)
        self._repeats = rospy.get_param('repeats', 3)
        self._baseline = rospy.get_param('baseline')
        self._sweep_params = rospy.get_param('sweep_params')
        self._param_grids = rospy.get_param('param_grids')
        self._speed_factor = rospy.get_param('speed_factor', 1.0)

        missing = [p for p in self._sweep_params if p not in self._baseline]
        if missing:
            raise RuntimeError('[TuningSweep] sweep_params %s missing from baseline' % missing)

        # arm is the innermost loop (not repeats), same interleaving fix
        # timing_sweep_orchestrator.py applied -- all arms for a given
        # (param, value) cell run back to back before repeating.
        self._combos = [
            (param, value, arm)
            for param in self._sweep_params
            for value in self._param_grids[param]
            for _ in range(self._repeats)
            for arm in ARMS
        ]
        rospy.loginfo(
            '[TuningSweep] %d trials queued (%d params x %d arms x %d repeats, '
            'grid sizes: %s).', len(self._combos), len(self._sweep_params), len(ARMS),
            self._repeats, {p: len(self._param_grids[p]) for p in self._sweep_params})

        results_dir = rospy.get_param('results_dir', _default_results_dir())
        self._tag_dir = os.path.join(results_dir, self._tag)
        self._refuse_if_already_completed()
        self._csv_dir = os.path.join(self._tag_dir, 'csv')
        os.makedirs(self._csv_dir, exist_ok=True)
        self._progress_path = os.path.join(self._csv_dir, 'progress.csv')
        self._done_ids = self._load_progress()

        self._trial_pub = rospy.Publisher('/haptiquad_nav_eval/tuning_trial_start',
                                          TuningTrialStart, queue_size=1, latch=True)
        self._override_pubs = {
            param: rospy.Publisher(topic, Float64, queue_size=1)
            for param, topic in OVERRIDE_TOPICS.items()
        }

        rospy.loginfo('[TuningSweep] Waiting for contact_probe_manager/set_probing_enabled…')
        rospy.wait_for_service('/contact_probe_manager/set_probing_enabled', timeout=30.0)
        self._set_probing = rospy.ServiceProxy(
            '/contact_probe_manager/set_probing_enabled', SetBool)

        self._spawner = ObjectSpawner()
        self._spawner.cleanup_stale(OBJECT_NAME_PREFIX)
        self._reset = SimReset(reset_settle_time=self._reset_settle_time)
        self._goals = GoalSender()
        self._pose = PoseCache()
        self._classification = ClassificationWatcher()
        self._health = SimHealthMonitor()

        # Optional speed-up -- see tuning_sweep.yaml's comment. restore() is
        # registered on shutdown FIRST (before apply()) so a crash/Ctrl-C
        # during startup still restores physics if apply() already changed it.
        self._rtf = RealTimeFactorController()
        rospy.on_shutdown(self._rtf.restore)
        if self._speed_factor != 1.0:
            self._rtf.apply(self._speed_factor)

        self._snapshot_config()

        self._prev_object_name = None
        rospy.sleep(1.0)  # let subscribers/publishers connect
        self._run()

    def _refuse_if_already_completed(self):
        plots_dir = os.path.join(self._tag_dir, 'plots')
        if os.path.isdir(plots_dir) and any(os.scandir(plots_dir)):
            raise RuntimeError(
                "[TuningSweep] Refusing to run: '%s' already contains a "
                "completed run (plots/ is non-empty). Pass a different "
                "experiment_tag, or move/delete that directory if you really "
                "want to redo it." % self._tag_dir)

    def _snapshot_config(self):
        path = os.path.join(self._tag_dir, 'tuning_sweep_config.yaml')
        try:
            import yaml
            params = {'baseline': self._baseline, 'param_grids': self._param_grids,
                     'sweep_params': self._sweep_params, 'speed_factor': self._speed_factor}
            with open(path, 'w') as f:
                yaml.safe_dump(params, f, sort_keys=True)
            rospy.loginfo('[TuningSweep] Saved sweep config to %s', path)
        except Exception as e:
            rospy.logwarn('[TuningSweep] Could not snapshot config: %s', e)

    # ── progress checkpointing ───────────────────────────────────────────────

    def _load_progress(self):
        if not os.path.exists(self._progress_path):
            return set()
        with open(self._progress_path) as f:
            return {int(row['trial_id']) for row in csv.DictReader(f)}

    def _mark_done(self, trial_id):
        new_file = not os.path.exists(self._progress_path)
        with open(self._progress_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['trial_id'])
            w.writerow([trial_id])

    # ── main loop ─────────────────────────────────────────────────────────────

    def _run(self):
        n_skipped = 0
        for trial_id, (param, value, arm) in enumerate(self._combos):
            if rospy.is_shutdown():
                break
            if trial_id in self._done_ids:
                n_skipped += 1
                continue
            self._run_one(trial_id, param, value, arm)
            self._mark_done(trial_id)
        if n_skipped:
            rospy.loginfo('[TuningSweep] Resumed: skipped %d already-completed trials.',
                          n_skipped)
        rospy.loginfo('[TuningSweep] Sweep complete (%d trials).', len(self._combos))
        self._rtf.restore()
        self._generate_plots()

    def _generate_plots(self):
        if not os.path.exists(os.path.join(self._csv_dir, 'summary.csv')):
            rospy.logwarn('[TuningSweep] No summary.csv written -- skipping plots.')
            return
        try:
            from haptiquad_nav_eval.tuning_plotting import generate_all as generate_all_plots
            written, errors = generate_all_plots(self._tag_dir)
        except Exception as e:
            rospy.logwarn('[TuningSweep] Plot generation failed entirely: %s', e)
            return
        for path in written:
            rospy.loginfo('[TuningSweep] Plot written: %s', path)
        for filename, e in errors:
            rospy.logwarn('[TuningSweep] Skipped plot %s: %s', filename, e)

    def _publish_param_overrides(self, param, value):
        """Publish every tunable parameter's value for this trial: baseline
        for all except `param`, which gets `value`. Always explicit, never
        left over from a previous trial's override."""
        for p in self._sweep_params:
            v = value if p == param else self._baseline[p]
            self._override_pubs[p].publish(Float64(data=float(v)))

    def _run_one(self, trial_id, param, value, arm):
        name = '%s%05d' % (OBJECT_NAME_PREFIX, trial_id)

        if self._prev_object_name is not None:
            self._spawner.delete(self._prev_object_name)
        self._reset.reset()

        # Probing is always enabled here -- there is no "classical" arm in
        # this experiment -- but set it explicitly every trial anyway, in
        # case a previous timing_sweep_orchestrator.py run left it disabled.
        self._set_probing(True)
        self._publish_param_overrides(param, value)

        pose = self._pose.get()
        if pose is None:
            rospy.logwarn('[TuningSweep] Trial %d: no /slam/pose yet after reset -- '
                          'waiting up to 10s.', trial_id)
            t0 = rospy.Time.now()
            while pose is None and (rospy.Time.now() - t0).to_sec() < 10.0 and not rospy.is_shutdown():
                rospy.sleep(0.1)
                pose = self._pose.get()
        if pose is None:
            rospy.logerr('[TuningSweep] Trial %d: never got /slam/pose -- skipping.', trial_id)
            self._prev_object_name = None
            return
        rx, ry, ryaw = pose

        if arm == 'movable':
            mass, mu = self._movable_mass, self._movable_mu
        else:
            mass, mu = self._solid_mass, self._solid_mu

        (ox, oy), (gx, gy) = plan_object_and_goal(
            rx, ry, ryaw, 0.0, 0.0, self._approach_distance, self._goal_overshoot)

        rospy.loginfo(
            '[TuningSweep] Trial %d: param=%s value=%s arm=%s mass=%.1fkg mu=%.2f '
            '(robot_origin=(%.2f,%.2f,%.1fdeg) obj=(%.2f,%.2f) goal=(%.2f,%.2f))',
            trial_id, param, value, arm, mass, mu, rx, ry, math.degrees(ryaw),
            ox, oy, gx, gy)

        self._spawner.spawn(name, self._geometry, mass, mu,
                            self._size_x, self._size_y, self._size_z, ox, oy)
        self._prev_object_name = name
        rospy.sleep(self._settle_time)

        msg = TuningTrialStart()
        msg.header.stamp = rospy.Time.now()   # doubles as t_goal_sent
        msg.trial_id = trial_id
        msg.experiment_tag = self._tag
        msg.trial_mode = arm
        msg.object_name = name
        msg.mass = mass
        msg.mu = mu
        msg.geometry = self._geometry
        msg.size_x, msg.size_y, msg.size_z = self._size_x, self._size_y, self._size_z
        msg.spawn_x, msg.spawn_y = ox, oy
        msg.goal_x, msg.goal_y = gx, gy
        msg.approach_angle = 0.0
        msg.robot_origin_x, msg.robot_origin_y, msg.robot_origin_yaw = rx, ry, ryaw
        msg.swept_param = param
        msg.swept_value = float(value)
        for p in self._sweep_params:
            v = value if p == param else self._baseline[p]
            if p == 'wrench_filter_window':
                setattr(msg, p, int(round(v)))
            else:
                setattr(msg, p, float(v))

        self._classification.arm()
        self._health.arm()
        self._trial_pub.publish(msg)
        self._goals.send(gx, gy, 0.0)

        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - t0).to_sec()
            if self._classification.resolved():
                break
            anomaly = self._health.anomaly()
            if anomaly is not None:
                rospy.logwarn('[TuningSweep] Trial %d (%s=%s, %s): %s -- abandoning trial '
                              'early (tuning_trial_logger.py will record it as anomalous).',
                              trial_id, param, value, arm, anomaly)
                break
            if not self._classification.saw_contact_probe and elapsed > self._no_contact_timeout:
                rospy.logwarn(
                    '[TuningSweep] Trial %d (%s=%s, %s): no contact within %.0fs -- '
                    'moving on (check probe_distance/probe_speed grid values).',
                    trial_id, param, value, arm, self._no_contact_timeout)
                break
            if elapsed > self._trial_timeout:
                rospy.logwarn('[TuningSweep] Trial %d (%s=%s, %s): overall timeout.',
                              trial_id, param, value, arm)
                break
            rospy.sleep(0.1)


if __name__ == '__main__':
    rospy.init_node('tuning_sweep_orchestrator')
    TuningSweepOrchestrator()
    rospy.spin()
