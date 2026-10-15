#!/usr/bin/env python3
"""
sweep_orchestrator.py — drives an unattended mass/friction/approach sweep
for HaptiQuad-Nav contact-classification evaluation.

Per trial:
  1. Delete the previous trial's object (if any) and fully reset the
     simulation (SimReset -- contact_probe_manager memory, costmaps,
     Gazebo world, state estimator) so every trial starts from the same
     clean state.
  2. Read the robot's post-reset pose from /slam/pose as this trial's
     origin, and spawn the object relative to it (approach_angle /
     approach_offset / approach_distance from sweep.yaml).
  3. Publish TrialStart with full ground-truth metadata -- the only
     handoff to trial_logger.py -- then immediately send the approach goal
     via plain /move_base_simple/goal (the same public interface RViz's
     "2D Nav Goal" and contact_probe_manager's own `_last_goal` tracking
     use).
  4. Wait for a full CONTACT_PROBE -> NORMAL_NAV state cycle (read from
     /haptiquad_nav/state), with timeouts as a safety net.

Does NOT compute or store any per-trial results itself -- trial_logger.py,
listening independently on the same topics, is the single source of truth
for logged data. Progress is checkpointed to results_dir so a crashed
sweep resumes instead of re-running completed trials.

Requires: haptiquad_anymal_bringup/launch/sim_eval.launch (empty world) and
haptiquad_nav's navigation.launch already running.
"""
import csv
import math
import os

import rospkg
import rospy

from haptiquad_nav_eval.trial_core import (ObjectSpawner, SimReset, GoalSender, PoseCache,
                                           StateWatcher, SimHealthMonitor,
                                           RealTimeFactorController, plan_object_and_goal)
from haptiquad_nav_eval.plotting import generate_all as generate_all_plots
from haptiquad_nav_eval.msg import TrialStart


def _default_results_dir():
    # Hidden folder inside the package itself (not the container's home
    # dir) so it lands under the host bind-mount (./haptiquad_nav_ws) and
    # is visible outside Docker. See package .gitignore -- .results/ is
    # deliberately excluded from git.
    return os.path.join(rospkg.RosPack().get_path('haptiquad_nav_eval'), '.results')


OBJECT_NAME_PREFIX = 'evalobj_'


class SweepOrchestrator:

    def __init__(self):
        self._tag = rospy.get_param('experiment_tag', 'sweep')
        self._approach_distance = rospy.get_param('approach_distance', 3.0)
        self._goal_overshoot = rospy.get_param('goal_overshoot', 1.5)
        self._settle_time = rospy.get_param('settle_time', 1.0)
        self._reset_settle_time = rospy.get_param('reset_settle_time', 1.5)
        self._no_contact_timeout = rospy.get_param('no_contact_timeout', 25.0)
        self._trial_timeout = rospy.get_param('trial_timeout', 60.0)
        self._repeats = rospy.get_param('repeats', 1)
        self._speed_factor = rospy.get_param('speed_factor', 1.0)

        geoms = rospy.get_param('geometries')
        masses = rospy.get_param('sweep/mass_kg')
        mus = rospy.get_param('sweep/mu')
        angles_deg = rospy.get_param('sweep/approach_angle_deg', [0.0])
        offsets = rospy.get_param('sweep/approach_offset_m', [0.0])

        self._combos = [
            (g, m, mu, math.radians(a), off)
            for g in geoms
            for m in masses
            for mu in mus
            for a in angles_deg
            for off in offsets
            for _ in range(self._repeats)
        ]
        rospy.loginfo('[SweepOrchestrator] %d trials queued.', len(self._combos))

        results_dir = rospy.get_param('results_dir', _default_results_dir())
        self._tag_dir = os.path.join(results_dir, self._tag)
        self._refuse_if_already_completed()  # before touching anything on disk
        self._csv_dir = os.path.join(self._tag_dir, 'csv')
        os.makedirs(self._csv_dir, exist_ok=True)
        self._progress_path = os.path.join(self._csv_dir, 'progress.csv')
        self._done_ids = self._load_progress()

        self._trial_pub = rospy.Publisher('/haptiquad_nav_eval/trial_start',
                                          TrialStart, queue_size=1, latch=True)
        self._spawner = ObjectSpawner()
        self._spawner.cleanup_stale(OBJECT_NAME_PREFIX)  # residue from a prior crashed run
        self._reset = SimReset(reset_settle_time=self._reset_settle_time)
        self._goals = GoalSender()
        self._pose = PoseCache()
        self._state = StateWatcher()
        self._health = SimHealthMonitor()

        # Optional speed-up -- see sweep.yaml's speed_factor comment.
        # restore() is registered on shutdown FIRST so a crash/Ctrl-C during
        # startup still restores physics if apply() already changed it.
        self._rtf = RealTimeFactorController()
        rospy.on_shutdown(self._rtf.restore)
        if self._speed_factor != 1.0:
            self._rtf.apply(self._speed_factor)

        self._snapshot_contact_probe_config()

        self._prev_object_name = None
        rospy.sleep(1.0)  # let subscribers/publishers connect
        self._run()

    def _refuse_if_already_completed(self):
        """A tag_dir whose plots/ already has content means a PRIOR run under
        this tag finished completely (_generate_plots() only runs at the very
        end of _run()) -- refuse to start rather than silently mixing a new
        run's trials into it or clobbering its results. An incomplete run
        (no plots yet, e.g. a crashed sweep) has no plots/ content and is
        still resumed normally via progress.csv, as before."""
        plots_dir = os.path.join(self._tag_dir, 'plots')
        if os.path.isdir(plots_dir) and any(os.scandir(plots_dir)):
            raise RuntimeError(
                "[SweepOrchestrator] Refusing to run: '%s' already contains a "
                "completed run (plots/ is non-empty). Pass a different "
                "experiment_tag, or move/delete that directory if you really "
                "want to redo it." % self._tag_dir)

    def _snapshot_contact_probe_config(self):
        """Save the live contact_probe_manager parameters actually in effect
        for this run, so a results folder is self-documenting even if
        contact_probe.yaml is edited again before anyone looks at it."""
        path = os.path.join(self._tag_dir, 'contact_probe_manager_config.yaml')
        try:
            import yaml
            params = rospy.get_param('/contact_probe_manager')
        except Exception as e:
            rospy.logwarn('[SweepOrchestrator] Could not snapshot contact_probe_manager '
                          'config (is navigation.launch running?): %s', e)
            return
        with open(path, 'w') as f:
            yaml.safe_dump(params, f, sort_keys=True)
        rospy.loginfo('[SweepOrchestrator] Saved contact_probe_manager config to %s', path)

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
        for trial_id, (geom, mass, mu, angle, offset) in enumerate(self._combos):
            if rospy.is_shutdown():
                break
            if trial_id in self._done_ids:
                n_skipped += 1
                continue
            self._run_one(trial_id, geom, mass, mu, angle, offset)
            self._mark_done(trial_id)
        if n_skipped:
            rospy.loginfo('[SweepOrchestrator] Resumed: skipped %d already-completed trials.',
                          n_skipped)
        rospy.loginfo('[SweepOrchestrator] Sweep complete (%d trials).', len(self._combos))
        self._rtf.restore()
        self._generate_plots()

    def _generate_plots(self):
        # Best-effort: a plotting failure (e.g. zero completed trials, or a
        # results directory from an older schema) should never make the
        # sweep itself look like it failed -- the CSVs are already safely
        # on disk regardless. Re-run scripts/plot_results.py by hand any
        # time to retry.
        if not os.path.exists(os.path.join(self._csv_dir, 'summary.csv')):
            rospy.logwarn('[SweepOrchestrator] No summary.csv written -- skipping plots.')
            return
        try:
            written, errors = generate_all_plots(self._tag_dir)
        except Exception as e:
            rospy.logwarn('[SweepOrchestrator] Plot generation failed entirely: %s', e)
            return
        for path in written:
            rospy.loginfo('[SweepOrchestrator] Plot written: %s', path)
        for filename, e in errors:
            rospy.logwarn('[SweepOrchestrator] Skipped plot %s: %s', filename, e)

    def _run_one(self, trial_id, geom, mass, mu, angle, offset):
        name = '%s%05d' % (OBJECT_NAME_PREFIX, trial_id)

        if self._prev_object_name is not None:
            self._spawner.delete(self._prev_object_name)
        self._reset.reset()

        pose = self._pose.get()
        if pose is None:
            rospy.logwarn('[SweepOrchestrator] Trial %d: no /slam/pose yet after reset -- '
                          'waiting up to 10s.', trial_id)
            t0 = rospy.Time.now()
            while pose is None and (rospy.Time.now() - t0).to_sec() < 10.0 and not rospy.is_shutdown():
                rospy.sleep(0.1)
                pose = self._pose.get()
        if pose is None:
            rospy.logerr('[SweepOrchestrator] Trial %d: never got /slam/pose -- skipping.',
                         trial_id)
            self._prev_object_name = None
            return
        rx, ry, ryaw = pose

        (ox, oy), (gx, gy) = plan_object_and_goal(
            rx, ry, ryaw, angle, offset, self._approach_distance, self._goal_overshoot)

        rospy.loginfo(
            '[SweepOrchestrator] Trial %d: %s mass=%.1fkg mu=%.2f angle=%.0fdeg offset=%.2fm '
            '(robot_origin=(%.2f,%.2f,%.1fdeg) obj=(%.2f,%.2f))',
            trial_id, geom['name'], mass, mu, math.degrees(angle), offset,
            rx, ry, math.degrees(ryaw), ox, oy)

        self._spawner.spawn(name, geom['name'], mass, mu,
                            geom['size_x'], geom['size_y'], geom['size_z'], ox, oy)
        self._prev_object_name = name
        rospy.sleep(self._settle_time)

        msg = TrialStart()
        msg.header.stamp = rospy.Time.now()   # doubles as t_goal_sent
        msg.trial_id = trial_id
        msg.experiment_tag = self._tag
        msg.object_name = name
        msg.mass = mass
        msg.mu = mu
        msg.geometry = geom['name']
        msg.size_x, msg.size_y, msg.size_z = geom['size_x'], geom['size_y'], geom['size_z']
        msg.spawn_x, msg.spawn_y = ox, oy
        msg.approach_angle = angle
        msg.approach_offset = offset
        msg.robot_origin_x, msg.robot_origin_y, msg.robot_origin_yaw = rx, ry, ryaw

        self._state.reset_edges()
        self._health.arm()
        self._trial_pub.publish(msg)
        self._goals.send(gx, gy, angle)

        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - t0).to_sec()
            if self._state.saw_contact_probe and self._state.state == 'NORMAL_NAV':
                break
            anomaly = self._health.anomaly()
            if anomaly is not None:
                rospy.logwarn('[SweepOrchestrator] Trial %d: %s -- abandoning trial early '
                              '(trial_logger.py will record it as anomalous).',
                              trial_id, anomaly)
                break
            if not self._state.saw_contact_probe and elapsed > self._no_contact_timeout:
                rospy.logwarn(
                    '[SweepOrchestrator] Trial %d: no contact within %.0fs -- moving on '
                    '(check approach_distance / object placement).',
                    trial_id, self._no_contact_timeout)
                break
            if elapsed > self._trial_timeout:
                rospy.logwarn('[SweepOrchestrator] Trial %d: overall timeout.', trial_id)
                break
            rospy.sleep(0.1)


if __name__ == '__main__':
    rospy.init_node('sweep_orchestrator')
    SweepOrchestrator()
