#!/usr/bin/env python3
"""
timing_sweep_orchestrator.py — drives the lateral-length TIME-COMPARISON
sweep: for each length in timing_sweep.yaml's lengths_m, runs 3 trials with
mass/mu held fixed --

  classical — contact_probe_manager's probing bypassed
              (~set_probing_enabled false): pure move_base circumvent
              navigation, as if the obstacle were a priori known and the
              robot never touched anything.
  movable   — probing enabled; object mass/mu chosen to be pushable.
  solid     — probing enabled; object mass/mu chosen to be immovable.

Completely independent from sweep_orchestrator.py (the mass/mu classification
sweep): separate config (timing_sweep.yaml), separate message type
(TimingTrialStart), separate results directory, separate progress
checkpointing. Both scripts may run against the same long-lived
navigation.launch stack (contact_probe_manager, move_base) but never at the
same time -- they'd fight over the same robot/goal.

Per trial:
  1. Delete the previous object (if any) and fully reset the simulation
     (SimReset), same as sweep_orchestrator.py.
  2. Explicitly set contact_probe_manager's probing bypass for THIS trial's
     arm (~set_probing_enabled) -- always set explicitly, never toggled,
     so a resumed/crashed run can never inherit the wrong mode from a
     previous trial.
  3. Publish this trial's exclusion/refine-radius overrides, scaled to the
     object's length -- contact_probe_manager's solid_excl_radius/
     movable_excl_radius/solid_refine_capture_radius are fixed-radius
     circles around a single contact point (see contact_probe.yaml); a
     radius sized for the ~0.5m test-world cubes will under-cover a long
     obstacle's true footprint. See _scale_radii().
  4. Spawn the object, publish TimingTrialStart, send the goal via plain
     /move_base_simple/goal (must stay the same public interface
     contact_probe_manager's _last_goal tracking expects -- see
     trial_core.py's GoalSender).
  5. Wait for GoalCompletionWatcher.reached() (ground-truth position at the
     goal, settled, see trial_core.py) or trial_timeout, with a sanity
     warning if a movable/solid trial never even entered CONTACT_PROBE
     (check approach_distance/object placement).

Does NOT compute or store results itself -- timing_trial_logger.py,
listening independently, is the single source of truth for logged data.
"""
import csv
import math
import os

import rospkg
import rospy
from std_msgs.msg import Float64
from std_srvs.srv import SetBool
from geometry_msgs.msg import PolygonStamped, Point32

from haptiquad_nav_eval.trial_core import (ObjectSpawner, SimReset, GoalSender, PoseCache,
                                           StateWatcher, GoalCompletionWatcher,
                                           plan_object_and_goal)
from haptiquad_nav_eval.msg import TimingTrialStart

OBJECT_NAME_PREFIX = 'timingobj_'
ARMS = ('classical', 'movable', 'solid')


def _default_results_dir():
    return os.path.join(rospkg.RosPack().get_path('haptiquad_nav_eval'), '.results')


class TimingSweepOrchestrator:

    def __init__(self):
        self._tag = rospy.get_param('experiment_tag', 'timing_sweep')
        self._geometry = rospy.get_param('geometry', 'box')
        self._size_x = rospy.get_param('size_x', 0.5)
        self._size_z = rospy.get_param('size_z', 1.0)
        self._lengths = rospy.get_param('lengths_m')
        self._movable_mass = rospy.get_param('movable_mass_kg', 5.0)
        self._movable_mu = rospy.get_param('movable_mu', 0.1)
        self._solid_mass = rospy.get_param('solid_mass_kg', 200.0)
        self._solid_mu = rospy.get_param('solid_mu', 0.8)
        self._approach_distance = rospy.get_param('approach_distance', 3.0)
        self._goal_overshoot = rospy.get_param('goal_overshoot', 1.5)
        self._settle_time = rospy.get_param('settle_time', 1.0)
        self._reset_settle_time = rospy.get_param('reset_settle_time', 1.5)
        self._goal_tolerance_m = rospy.get_param('goal_tolerance_m', 0.30)
        self._goal_settle_time = rospy.get_param('goal_settle_time', 1.0)
        self._no_contact_timeout = rospy.get_param('no_contact_timeout', 25.0)
        self._trial_timeout = rospy.get_param('trial_timeout', 180.0)
        self._repeats = rospy.get_param('repeats', 3)
        self._excl_margin = rospy.get_param('excl_radius_margin_m', 0.3)

        # arm is the innermost loop (not repeats) so all three arms run back
        # to back for a given length before repeating -- e.g.
        # classical,movable,solid,classical,movable,solid,... rather than
        # three classical trials in a row followed by three movable, etc.
        self._combos = [
            (length, arm)
            for length in self._lengths
            for _ in range(self._repeats)
            for arm in ARMS
        ]
        rospy.loginfo('[TimingSweep] %d trials queued (%d lengths × %d arms × %d repeats).',
                      len(self._combos), len(self._lengths), len(ARMS), self._repeats)

        results_dir = rospy.get_param('results_dir', _default_results_dir())
        self._tag_dir = os.path.join(results_dir, self._tag)
        self._refuse_if_already_completed()
        self._csv_dir = os.path.join(self._tag_dir, 'csv')
        os.makedirs(self._csv_dir, exist_ok=True)
        self._progress_path = os.path.join(self._csv_dir, 'progress.csv')
        self._done_ids = self._load_progress()

        self._trial_pub = rospy.Publisher('/haptiquad_nav_eval/timing_trial_start',
                                          TimingTrialStart, queue_size=1, latch=True)
        self._solid_excl_pub = rospy.Publisher(
            '/contact_probe_manager/solid_excl_radius_override', Float64, queue_size=1)
        self._movable_excl_pub = rospy.Publisher(
            '/contact_probe_manager/movable_excl_radius_override', Float64, queue_size=1)
        self._refine_capture_pub = rospy.Publisher(
            '/contact_probe_manager/solid_refine_capture_radius_override', Float64, queue_size=1)
        self._inject_pub = rospy.Publisher(
            '/contact_probe_manager/inject_known_solid', PolygonStamped, queue_size=1)

        rospy.loginfo('[TimingSweep] Waiting for contact_probe_manager/set_probing_enabled…')
        rospy.wait_for_service('/contact_probe_manager/set_probing_enabled', timeout=30.0)
        self._set_probing = rospy.ServiceProxy(
            '/contact_probe_manager/set_probing_enabled', SetBool)

        self._spawner = ObjectSpawner()
        self._spawner.cleanup_stale(OBJECT_NAME_PREFIX)
        self._reset = SimReset(reset_settle_time=self._reset_settle_time)
        self._goals = GoalSender()
        self._pose = PoseCache()
        self._state = StateWatcher()
        self._completion = GoalCompletionWatcher(self._goal_tolerance_m, self._goal_settle_time)

        self._snapshot_contact_probe_config()

        self._prev_object_name = None
        rospy.sleep(1.0)  # let subscribers/publishers connect
        self._run()

    def _refuse_if_already_completed(self):
        plots_dir = os.path.join(self._tag_dir, 'plots')
        if os.path.isdir(plots_dir) and any(os.scandir(plots_dir)):
            raise RuntimeError(
                "[TimingSweep] Refusing to run: '%s' already contains a "
                "completed run (plots/ is non-empty). Pass a different "
                "experiment_tag, or move/delete that directory if you really "
                "want to redo it." % self._tag_dir)

    def _snapshot_contact_probe_config(self):
        path = os.path.join(self._tag_dir, 'contact_probe_manager_config.yaml')
        try:
            import yaml
            params = rospy.get_param('/contact_probe_manager')
        except Exception as e:
            rospy.logwarn('[TimingSweep] Could not snapshot contact_probe_manager '
                          'config (is navigation.launch running?): %s', e)
            return
        with open(path, 'w') as f:
            yaml.safe_dump(params, f, sort_keys=True)
        rospy.loginfo('[TimingSweep] Saved contact_probe_manager config to %s', path)

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
        for trial_id, (length, arm) in enumerate(self._combos):
            if rospy.is_shutdown():
                break
            if trial_id in self._done_ids:
                n_skipped += 1
                continue
            self._run_one(trial_id, length, arm)
            self._mark_done(trial_id)
        if n_skipped:
            rospy.loginfo('[TimingSweep] Resumed: skipped %d already-completed trials.',
                          n_skipped)
        rospy.loginfo('[TimingSweep] Sweep complete (%d trials).', len(self._combos))
        self._generate_plots()

    def _generate_plots(self):
        if not os.path.exists(os.path.join(self._csv_dir, 'summary.csv')):
            rospy.logwarn('[TimingSweep] No summary.csv written -- skipping plots.')
            return
        try:
            from haptiquad_nav_eval.timing_plotting import generate_all as generate_all_plots
            written, errors = generate_all_plots(self._tag_dir)
        except Exception as e:
            rospy.logwarn('[TimingSweep] Plot generation failed entirely: %s', e)
            return
        for path in written:
            rospy.loginfo('[TimingSweep] Plot written: %s', path)
        for filename, e in errors:
            rospy.logwarn('[TimingSweep] Skipped plot %s: %s', filename, e)

    def _scale_radii(self, length):
        """Scale contact_probe_manager's solid_excl_radius/
        solid_refine_capture_radius to cover this trial's (stationary)
        object -- half the lateral length plus a fixed margin comfortably
        bounds a point-+-radius approximation of the object regardless of
        where along its face the robot happens to make contact.

        Deliberately NOT used for movable_excl_radius -- see
        _movable_excl_radius() below, a solid never moves once classified so
        its own footprint is the only thing that radius needs to cover, but
        a movable object's exclusion radius needs to cover something else
        entirely."""
        return length / 2.0 + self._excl_margin

    def _movable_excl_radius(self, length):
        """movable_excl_radius must cover BOTH the object's own lateral
        extent AND the full remaining push-through distance to the goal --
        NOT just the object's footprint like _scale_radii() above.

        While the robot is actively pushing a movable object, the object
        stays within contact_probe_manager's lidar blind spot
        (cloud_range_min, ~0.55m) continuously, so _movable_obstacles'
        tracked centroid is NEVER updated during the push (see
        contact_probe_manager.py _cloud_cb's blind-spot skip) -- it stays
        frozen wherever it was seeded at MOVABLE classification. If the
        exclusion radius only covers the object's own length (as a naive
        reuse of _scale_radii() first did here), the object's true,
        continuously-advancing position outruns the frozen exclusion zone
        partway through the push, reappears as the "nearest obstacle", and
        contact_probe_manager re-triggers SLOW_APPROACH/CONTACT_PROBE
        against the very object it already classified MOVABLE -- sending a
        fresh probe goal ever further forward each time. That is exactly
        the "kept going forward past the goal" failure this fixes: since
        object and goal are colinear in this experiment, the remaining push
        distance is bounded by goal_overshoot, so that (not object length)
        is the dominant term."""
        return max(length / 2.0 + self._excl_margin,
                  self._goal_overshoot + self._excl_margin)

    def _publish_known_solid(self, cx, cy, size_x, size_y):
        """Classical arm only -- tells contact_probe_manager's
        ~inject_known_solid the box's ground-truth footprint (axis-aligned,
        since approach_angle/box yaw are both fixed at 0 for this
        experiment) so the global costmap marks it BEFORE the goal is sent,
        without ever running the probe/classify/backup pipeline. Without
        this, ~set_probing_enabled(false) alone leaves the global planner
        with no idea the obstacle exists -- see the module docstring."""
        hx, hy = size_x / 2.0, size_y / 2.0
        corners = [(cx - hx, cy - hy), (cx + hx, cy - hy),
                  (cx + hx, cy + hy), (cx - hx, cy + hy)]
        msg = PolygonStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.polygon.points = [Point32(x=x, y=y, z=0.0) for x, y in corners]
        self._inject_pub.publish(msg)

    def _run_one(self, trial_id, length, arm):
        name = '%s%05d' % (OBJECT_NAME_PREFIX, trial_id)

        if self._prev_object_name is not None:
            self._spawner.delete(self._prev_object_name)
        self._reset.reset()

        # Explicitly set (never toggle) so a resumed/crashed run can't
        # inherit the wrong bypass state from whatever the previous trial was.
        self._set_probing(not (arm == 'classical'))

        radius = self._scale_radii(length)
        movable_radius = self._movable_excl_radius(length)
        self._solid_excl_pub.publish(Float64(data=radius))
        self._movable_excl_pub.publish(Float64(data=movable_radius))
        self._refine_capture_pub.publish(Float64(data=radius))

        pose = self._pose.get()
        if pose is None:
            rospy.logwarn('[TimingSweep] Trial %d: no /slam/pose yet after reset -- '
                          'waiting up to 10s.', trial_id)
            t0 = rospy.Time.now()
            while pose is None and (rospy.Time.now() - t0).to_sec() < 10.0 and not rospy.is_shutdown():
                rospy.sleep(0.1)
                pose = self._pose.get()
        if pose is None:
            rospy.logerr('[TimingSweep] Trial %d: never got /slam/pose -- skipping.', trial_id)
            self._prev_object_name = None
            return
        rx, ry, ryaw = pose

        if arm == 'movable':
            mass, mu = self._movable_mass, self._movable_mu
        else:
            mass, mu = self._solid_mass, self._solid_mu   # solid AND classical

        (ox, oy), (gx, gy) = plan_object_and_goal(
            rx, ry, ryaw, 0.0, 0.0, self._approach_distance, self._goal_overshoot)

        rospy.loginfo(
            '[TimingSweep] Trial %d: arm=%s length=%.2fm mass=%.1fkg mu=%.2f '
            '(robot_origin=(%.2f,%.2f,%.1fdeg) obj=(%.2f,%.2f) goal=(%.2f,%.2f) '
            'solid_excl_r=%.2f movable_excl_r=%.2f)',
            trial_id, arm, length, mass, mu, rx, ry, math.degrees(ryaw),
            ox, oy, gx, gy, radius, movable_radius)

        self._spawner.spawn(name, self._geometry, mass, mu,
                            self._size_x, length, self._size_z, ox, oy)
        self._prev_object_name = name
        rospy.sleep(self._settle_time)

        if arm == 'classical':
            # Mark the costmap BEFORE the goal is sent -- see
            # _publish_known_solid's docstring. Without this the bypassed
            # state machine leaves the global planner with no idea the
            # object exists and the robot walks straight through it.
            self._publish_known_solid(ox, oy, self._size_x, length)
            rospy.sleep(0.2)  # let the costmap's obstacle_layer ingest it before planning

        msg = TimingTrialStart()
        msg.header.stamp = rospy.Time.now()   # doubles as t_goal_sent
        msg.trial_id = trial_id
        msg.experiment_tag = self._tag
        msg.trial_mode = arm
        msg.object_name = name
        msg.mass = mass
        msg.mu = mu
        msg.geometry = self._geometry
        msg.size_x, msg.size_y, msg.size_z = self._size_x, length, self._size_z
        msg.spawn_x, msg.spawn_y = ox, oy
        msg.goal_x, msg.goal_y = gx, gy
        msg.approach_angle = 0.0
        msg.robot_origin_x, msg.robot_origin_y, msg.robot_origin_yaw = rx, ry, ryaw

        self._state.reset_edges()
        self._completion.arm(gx, gy)
        self._trial_pub.publish(msg)
        self._goals.send(gx, gy, 0.0)

        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - t0).to_sec()
            if self._completion.reached():
                break
            if (arm != 'classical' and not self._state.saw_contact_probe and
                    elapsed > self._no_contact_timeout):
                rospy.logwarn(
                    '[TimingSweep] Trial %d (%s): no contact within %.0fs -- moving on '
                    '(check approach_distance / object placement).',
                    trial_id, arm, self._no_contact_timeout)
                break
            if elapsed > self._trial_timeout:
                rospy.logwarn('[TimingSweep] Trial %d (%s): overall timeout.', trial_id, arm)
                break
            rospy.sleep(0.1)


if __name__ == '__main__':
    rospy.init_node('timing_sweep_orchestrator')
    TimingSweepOrchestrator()
