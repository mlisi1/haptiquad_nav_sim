"""
trial_core.py — shared trial-execution primitives for the HaptiQuad-Nav
evaluation harness. Imported as haptiquad_nav_eval.trial_core by
sweep_orchestrator.py; not a node itself.

Lives under src/haptiquad_nav_eval/ (catkin_python_setup(), same pattern as
haptiquad_plot's src/haptiquad_plot/) rather than scripts/, specifically
because catkin_install_python's devel-space wrapper mechanism is only
correct for standalone-executable scripts: it wraps every file in that
DESTINATION list in an exec() shim meant to be run as __main__, which
silently breaks `from trial_core import X` (the wrapped module ends up with
none of its top-level names visible to the importer). Do not move this
file into scripts/ and catkin_install_python it "for convenience" -- that
exact mistake is what this comment is here to prevent.

Design constraint: nothing here ever touches the robot's Gazebo state
directly via /gazebo/set_model_state. The robot is repositioned between
trials by resetting the whole simulation (SimReset, below) back to its
launch-time spawn pose, then reading that pose back from /slam/pose --
never by teleporting a live robot mid-run.
"""
import math
import subprocess

import rospy
import tf.transformations
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
from std_msgs.msg import String
from std_srvs.srv import Empty as EmptySrv
from gazebo_msgs.srv import (SpawnModel, DeleteModel, GetWorldProperties,
                             GetPhysicsProperties, SetPhysicsProperties)


G = 9.81  # m/s^2, matches gravity_z in haptiquad_anymal_bringup/config/default.yaml


# ── SDF generation ──────────────────────────────────────────────────────────

def _box_inertia(m, sx, sy, sz):
    return (m / 12.0 * (sy**2 + sz**2),
            m / 12.0 * (sx**2 + sz**2),
            m / 12.0 * (sx**2 + sy**2))


def _cylinder_inertia(m, diameter, h):
    r = diameter / 2.0
    ixx_iyy = m / 12.0 * (3 * r**2 + h**2)
    izz = m / 2.0 * r**2
    return (ixx_iyy, ixx_iyy, izz)


def make_sdf(name, geometry, mass, mu, size_x, size_y, size_z):
    """Build a self-contained SDF model XML for /gazebo/spawn_sdf_model,
    matching the box/cylinder style already used in the checked-in test
    worlds (haptiquad_anymal_bringup/worlds/*.world) so eval objects behave
    identically to the pre-placed ones."""
    if geometry == 'box':
        ixx, iyy, izz = _box_inertia(mass, size_x, size_y, size_z)
        geom_xml = '<box><size>%s %s %s</size></box>' % (size_x, size_y, size_z)
    elif geometry == 'cylinder':
        ixx, iyy, izz = _cylinder_inertia(mass, size_x, size_z)
        geom_xml = ('<cylinder><radius>%s</radius><length>%s</length></cylinder>'
                    % (size_x / 2.0, size_z))
    else:
        raise ValueError('Unknown geometry %r (expected "box" or "cylinder")' % geometry)

    return '''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="%(name)s">
    <link name="link">
      <inertial>
        <mass>%(mass)s</mass>
        <inertia>
          <ixx>%(ixx)s</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>%(iyy)s</iyy><iyz>0</iyz>
          <izz>%(izz)s</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>%(geom_xml)s</geometry>
        <surface>
          <friction><ode><mu>%(mu)s</mu><mu2>%(mu)s</mu2></ode></friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>%(geom_xml)s</geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Green</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>''' % dict(name=name, mass=mass, mu=mu, ixx=ixx, iyy=iyy, izz=izz, geom_xml=geom_xml)


# ── Gazebo object spawn/delete (never touches the robot) ────────────────────

class ObjectSpawner:

    def __init__(self):
        rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=30.0)
        rospy.wait_for_service('/gazebo/delete_model', timeout=30.0)
        rospy.wait_for_service('/gazebo/get_world_properties', timeout=30.0)
        self._spawn = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self._delete = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self._get_world_properties = rospy.ServiceProxy(
            '/gazebo/get_world_properties', GetWorldProperties)

    def spawn(self, name, geometry, mass, mu, size_x, size_y, size_z, x, y, yaw=0.0):
        # /gazebo/reset_world does not delete dynamically-spawned models, only
        # resets existing ones' poses -- so a name collision here means a
        # stale leftover from a prior crashed/killed run is still in Gazebo.
        # Clear it quietly first rather than fail the whole sweep on it; the
        # common case (a name never used before) is silent and free.
        self._delete_quiet(name)

        sdf = make_sdf(name, geometry, mass, mu, size_x, size_y, size_z)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = size_z / 2.0 + 0.01   # tiny clearance above ground
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        pose.orientation = Quaternion(*q)
        resp = self._spawn(name, sdf, '', pose, 'world')
        if not resp.success:
            raise RuntimeError('spawn_sdf_model failed for %s: %s'
                                % (name, resp.status_message))

    def delete(self, name):
        try:
            resp = self._delete(name)
            if not resp.success:
                rospy.logwarn('[EvalSpawner] delete_model(%s): %s', name, resp.status_message)
        except rospy.ServiceException as e:
            rospy.logwarn('[EvalSpawner] delete_model(%s) failed (may already be gone): %s',
                          name, e)

    def _delete_quiet(self, name):
        try:
            self._delete(name)
        except rospy.ServiceException:
            pass

    def cleanup_stale(self, prefix):
        """Delete any leftover models whose name starts with `prefix` --
        residue from a prior crashed/killed sweep that /gazebo/reset_world
        never removed. Call once at orchestrator/calibrator startup so a
        resumed run doesn't collide with its own past self."""
        try:
            resp = self._get_world_properties()
        except rospy.ServiceException as e:
            rospy.logwarn('[EvalSpawner] get_world_properties failed, '
                          'skipping stale-model cleanup: %s', e)
            return
        stale = [n for n in resp.model_names if n.startswith(prefix)]
        for n in stale:
            rospy.logwarn('[EvalSpawner] Deleting stale leftover model from a prior run: %s', n)
            self.delete(n)


# ── Motion-controller recovery (post-reset / post-tip-over) ──────────────────

class MotionRecovery:
    """Re-clears the ANYmal e-stop and resends the trekker/walk goal -- the
    exact two calls haptiquad_anymal_bringup/scripts/init_motion.py makes at
    cold start.

    Why this is needed here too: /gazebo/reset_world (SimReset, below)
    teleports rigid-body poses/velocities in a single physics step but does
    NOT resync the legged robot's own walk-controller gait phase -- the
    controller still expects its feet where they were an instant before the
    teleport, relative to whatever ground is now under the new pose. That
    mismatch is what actually produced the tip-over observed mid-sweep (base
    roll spiking to ~99 degrees a fraction of a second into the very next
    trial after a reset). Once tipped over, motion_control_manager appears
    to fault into the same protective e-stop state it starts in cold --
    after which every subsequent trial's goal is silently ignored (the robot
    never moves, but nothing in haptiquad_nav is actually broken) until this
    exact recovery sequence runs again. subprocess is used for the walk-goal
    publish for the same reason init_motion.py uses it: avoids importing the
    proprietary anymal_motion_control_msgs package at the Python level.

    Best-effort and safe to call unconditionally (idempotent if the robot
    never actually faulted) -- call it after every SimReset.reset() as cheap
    insurance, not only when SimHealthMonitor has already flagged a tip-over.
    """

    _TREKKER_WALK_GOAL = """\
header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
goal_id:
  stamp: {secs: 0, nsecs: 0}
  id: ''
goal:
  motion:
    motion_controller: 'trekker'
    reference_type: {data: 2}
    operation_mode: 'walk'
  motion_parameters: ''"""

    def __init__(self):
        self._clear_estop = None
        try:
            rospy.wait_for_service('/motion_control_manager/clear_emergency_stop', timeout=10.0)
            self._clear_estop = rospy.ServiceProxy(
                '/motion_control_manager/clear_emergency_stop', EmptySrv)
        except rospy.ROSException:
            rospy.logwarn('[MotionRecovery] /motion_control_manager/clear_emergency_stop '
                          'not available -- automatic post-reset/tip-over recovery is '
                          'disabled; a tip-over will permanently freeze the rest of the sweep.')

    def recover(self):
        if self._clear_estop is not None:
            try:
                self._clear_estop()
            except rospy.ServiceException as e:
                rospy.logwarn('[MotionRecovery] clear_emergency_stop failed: %s', e)
        result = subprocess.run(
            ['rostopic', 'pub', '-1',
             '/motion_control_manager/execute_motion/goal',
             'anymal_motion_control_msgs/ExecuteMotionActionGoal',
             self._TREKKER_WALK_GOAL],
            capture_output=True, text=True)
        if result.returncode != 0:
            rospy.logwarn('[MotionRecovery] rostopic pub failed:\n%s', result.stderr)


# ── Simulation reset between trials ──────────────────────────────────────────

class SimReset:
    """Resets the world + robot state estimate between trials.

    UNVERIFIED end-to-end (built from static config inspection, not a live
    test -- see the comments below and re-check /slam/pose visually before
    trusting a full unattended sweep on this).

    Sequence, in order:
      1. contact_probe_manager's own ~reset service -- clears its
         accumulated _solid_obstacles / _movable_obstacles memory, which
         otherwise persists for the node's entire lifetime and leaks into
         the next trial after a world reset.
      2. /move_base/clear_costmaps -- wipes the actual costmap grids
         (contact_probe_manager's solid_obstacle_layer marks are permanent
         -- clearing: false -- specifically so they survive incidental
         costmap resets during normal operation; here we WANT them gone).
      3. /gazebo/reset_world -- resets model poses/velocities (including
         the robot) back to their spawn pose. Deliberately NOT
         reset_simulation, which also resets sim time and would be far more
         likely to upset time-stamped TF/estimator state.
      4. /state_estimator/reset -- the proprietary stack documents a reset
         cascade for this in default.yaml's `signal_relay.reset_routes`:
         the state estimator publishes /state_estimator/reset_notification
         after resetting, which signal_relay is wired to forward into
         /point_cloud_odometry/reset, which itself cascades further into
         /localization_manager/point_cloud_odometry_connector/reset and
         /elevation_mapping/clear_map. This is presumably the SUPPORTED way
         to tell the estimation stack "the robot was relocated" -- as
         opposed to a bare Gazebo-side pose reset with no estimator
         resync, which is the likely cause of the problems reported from
         prior manual attempts at resetting this sim.
      5. MotionRecovery.recover() -- re-clears the e-stop and resends the
         trekker/walk goal. reset_world (step 3) does not resync the walk
         controller's gait phase, which can and does tip the robot over on
         the very next trial; once fallen, motion_control_manager appears to
         fault the same way it does cold, silently ignoring every later
         goal. This step is cheap insurance against that, run on every
         reset rather than only after a detected tip-over.
    """

    def __init__(self, reset_settle_time=1.5):
        self._reset_settle_time = reset_settle_time

        rospy.wait_for_service('/gazebo/reset_world', timeout=30.0)
        self._gazebo_reset = rospy.ServiceProxy('/gazebo/reset_world', EmptySrv)

        self._state_estimator_reset = None
        try:
            rospy.wait_for_service('/state_estimator/reset', timeout=10.0)
            self._state_estimator_reset = rospy.ServiceProxy('/state_estimator/reset', EmptySrv)
        except rospy.ROSException:
            rospy.logwarn('[SimReset] /state_estimator/reset not available -- '
                          'skipping estimator resync. Localization may drift '
                          'across trials.')

        self._clear_costmaps = None
        try:
            rospy.wait_for_service('/move_base/clear_costmaps', timeout=10.0)
            self._clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', EmptySrv)
        except rospy.ROSException:
            rospy.logwarn('[SimReset] /move_base/clear_costmaps not available.')

        self._cpm_reset = None
        try:
            rospy.wait_for_service('/contact_probe_manager/reset', timeout=10.0)
            self._cpm_reset = rospy.ServiceProxy('/contact_probe_manager/reset', EmptySrv)
        except rospy.ROSException:
            rospy.logwarn('[SimReset] /contact_probe_manager/reset not available -- '
                          'is navigation.launch running? Obstacle memory will '
                          'leak between trials without it.')

        self._motion_recovery = MotionRecovery()

    def reset(self):
        if self._cpm_reset is not None:
            try:
                self._cpm_reset()
            except rospy.ServiceException as e:
                rospy.logwarn('[SimReset] contact_probe_manager reset failed: %s', e)
        if self._clear_costmaps is not None:
            try:
                self._clear_costmaps()
            except rospy.ServiceException as e:
                rospy.logwarn('[SimReset] clear_costmaps failed: %s', e)

        self._gazebo_reset()

        if self._state_estimator_reset is not None:
            try:
                self._state_estimator_reset()
            except rospy.ServiceException as e:
                rospy.logwarn('[SimReset] state_estimator reset failed: %s', e)

        self._motion_recovery.recover()

        rospy.sleep(self._reset_settle_time)


# ── Robot repositioning via ordinary goals (no Gazebo teleport) ─────────────

class GoalSender:
    """Publishes plain PoseStamped goals on /move_base_simple/goal."""

    def __init__(self):
        self._pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        rospy.sleep(0.5)  # let the publisher connect before the first send

    def send(self, x, y, yaw, frame_id='map'):
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = x
        msg.pose.position.y = y
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        msg.pose.orientation = Quaternion(*q)
        self._pub.publish(msg)


def plan_object_and_goal(robot_x, robot_y, robot_yaw, approach_angle, approach_offset,
                         approach_distance, goal_overshoot):
    """Given the robot's pose right after a reset, compute where to spawn
    the trial object and where to send the approach goal -- both (x, y)
    map-frame tuples, with the goal also carrying `approach_angle` as its
    yaw. approach_angle/approach_offset are absolute map-frame quantities
    (not relative to robot_yaw), since the robot's reset pose is fixed and
    known -- see robot_yaw's docstring note in TrialStart.msg.

    approach_offset: lateral perpendicular offset (m) of the approach line
        from the object's center (positive = left of the approach direction).
    """
    ca, sa = math.cos(approach_angle), math.sin(approach_angle)
    px, py = -sa, ca  # unit vector perpendicular to the approach direction
    ox = robot_x + approach_distance * ca + approach_offset * px
    oy = robot_y + approach_distance * sa + approach_offset * py
    goal_x = robot_x + (approach_distance + goal_overshoot) * ca + approach_offset * px
    goal_y = robot_y + (approach_distance + goal_overshoot) * sa + approach_offset * py
    return (ox, oy), (goal_x, goal_y)


class PoseCache:
    """Latest (x, y, yaw) from /slam/pose -- the same pose source
    contact_probe_manager itself uses."""

    def __init__(self):
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self.pose = None  # (x, y, yaw)
        rospy.Subscriber('/slam/pose', PoseWithCovarianceStamped, self._cb, queue_size=1)

    def _cb(self, msg):
        p = msg.pose.pose
        _, _, yaw = tf.transformations.euler_from_quaternion(
            [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
        self.pose = (p.position.x, p.position.y, yaw)

    def get(self):
        return self.pose


class GoalCompletionWatcher:
    """Detects genuine goal-reached completion for
    timing_sweep_orchestrator.py / timing_trial_logger.py -- NOT used by the
    mass/mu sweep (StateWatcher below is unaffected and unchanged).

    "Reached" is defined by ground-truth position (within goal_tolerance_m of
    the commanded goal, continuously, for goal_settle_time seconds, while the
    state machine is back in NORMAL_NAV) rather than by watching
    /move_base/status for a terminal code: contact_probe_manager resends the
    ORIGINAL goal as a brand-new actionlib goal every time it returns to
    NORMAL_NAV (after CONTACT_PROBE/CLASSIFYING/BACKING_UP), and
    /move_base/status only ever exposes whatever goal is currently tracked --
    a naive "first terminal status seen after NORMAL_NAV" check races the
    just-preempted probe/backup goal's own transient status update, which can
    arrive before the resent goal's real result. Position-based detection
    sidesteps that race entirely and measures what "reached the goal" means
    physically, which is also the quantity the timing experiment actually
    cares about.
    """

    def __init__(self, goal_tolerance_m, goal_settle_time):
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self._tol2 = goal_tolerance_m ** 2
        self._settle = goal_settle_time
        self.state = None
        self._goal = None              # (x, y), None until armed
        self._within_tol_since = None
        rospy.Subscriber('/haptiquad_nav/state', String, self._state_cb, queue_size=10)
        rospy.Subscriber('/slam/pose', PoseWithCovarianceStamped, self._pose_cb, queue_size=1)

    def arm(self, goal_x, goal_y):
        """Call once per trial, right before/after sending that trial's goal."""
        self._goal = (goal_x, goal_y)
        self._within_tol_since = None

    def _state_cb(self, msg):
        self.state = msg.data

    def _pose_cb(self, msg):
        if self._goal is None:
            return
        p = msg.pose.pose.position
        gx, gy = self._goal
        d2 = (p.x - gx) ** 2 + (p.y - gy) ** 2
        if d2 <= self._tol2 and self.state == 'NORMAL_NAV':
            if self._within_tol_since is None:
                self._within_tol_since = rospy.Time.now()
        else:
            self._within_tol_since = None

    def reached(self):
        return (self._within_tol_since is not None and
                (rospy.Time.now() - self._within_tol_since).to_sec() >= self._settle)


class ClassificationWatcher:
    """Detects the CLASSIFYING state's EXIT -- to NORMAL_NAV (MOVABLE) or to
    BACKING_UP (SOLID) -- for tuning_sweep_orchestrator.py /
    tuning_trial_logger.py (the contact-probe parameter-tuning harness).

    Deliberately different from both other trial-boundary primitives here:
    trial_logger.py (the mass/mu sweep) waits for a full return to
    NORMAL_NAV, which for a SOLID verdict means waiting out the entire
    BACKING_UP maneuver; GoalCompletionWatcher (the lateral-length timing
    sweep) waits for the real goal to be reached. The tuning sweep only
    cares about classification TIME, so waiting for anything past
    classify_exit would only burn wall-clock time on data it never uses --
    this fires the instant the decision is made, regardless of arm.

    Like GoalCompletionWatcher, both the orchestrator and the logger each
    create their OWN instance (independent subscriptions to the same
    topic) rather than sharing one across nodes.
    """

    def __init__(self):
        self.state = None
        self._prev_state = None
        self.saw_contact_probe = False
        self._t_probe_entry = None
        self._t_classify_entry = None
        self._t_classify_exit = None
        self._exited_to = None   # 'NORMAL_NAV' (movable) | 'BACKING_UP' (solid) | None
        rospy.Subscriber('/haptiquad_nav/state', String, self._cb, queue_size=10)

    def arm(self):
        """Call once per trial, right before/after sending that trial's goal."""
        self.saw_contact_probe = False
        self._t_probe_entry = None
        self._t_classify_entry = None
        self._t_classify_exit = None
        self._exited_to = None

    def _cb(self, msg):
        self._prev_state = self.state
        self.state = msg.data
        if msg.data == 'CONTACT_PROBE' and self._prev_state != 'CONTACT_PROBE':
            self.saw_contact_probe = True
            if self._t_probe_entry is None:
                self._t_probe_entry = rospy.Time.now().to_sec()
        if msg.data == 'CLASSIFYING' and self._prev_state != 'CLASSIFYING':
            self._t_classify_entry = rospy.Time.now().to_sec()
        if self._prev_state == 'CLASSIFYING' and msg.data != 'CLASSIFYING':
            self._t_classify_exit = rospy.Time.now().to_sec()
            self._exited_to = msg.data

    def resolved(self):
        return self._exited_to is not None

    @property
    def t_probe_entry(self):
        return self._t_probe_entry

    @property
    def t_classify_entry(self):
        return self._t_classify_entry

    @property
    def t_classify_exit(self):
        return self._t_classify_exit

    @property
    def exited_to(self):
        return self._exited_to


class RealTimeFactorController:
    """Optional, best-effort Gazebo speed-up for tuning_sweep_orchestrator.py
    (EXPERIMENTAL -- see tuning_sweep.yaml's speed_factor comment).

    Only scales max_update_rate; time_step, gravity, and every ODE solver
    setting are read back and re-sent unchanged, so simulated
    dynamics/physics-step granularity is NOT affected -- only how many
    wall-clock seconds it takes to compute a fixed number of steps. Since
    navigation.launch sets /use_sim_time:true, every timestamp this harness
    records (rospy.Time.now()) is simulated time, so classification-time
    metrics are unaffected by whatever real-time factor is actually
    achieved -- this only shortens how long a human waits for the sweep to
    finish. If the CPU can't sustain the requested rate, Gazebo just falls
    back to whatever it can sustain -- a target, not a guarantee.

    NOT verified against the proprietary ANYmal controller/estimator
    stack's own internal timing assumptions. If stability/estimator-health
    metrics get visibly worse only when speed_factor > 1, that's a sign the
    low-level stack doesn't tolerate it well -- drop back to 1.0.
    """

    def __init__(self):
        self._orig_max_update_rate = None
        self._get = None
        self._set = None
        try:
            rospy.wait_for_service('/gazebo/get_physics_properties', timeout=10.0)
            rospy.wait_for_service('/gazebo/set_physics_properties', timeout=10.0)
            self._get = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
            self._set = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        except rospy.ROSException as e:
            rospy.logwarn('[RealTimeFactorController] Gazebo physics services '
                          'unavailable -- speed_factor will be ignored: %s', e)

    def apply(self, speed_factor):
        if self._get is None or speed_factor is None or speed_factor == 1.0:
            return
        try:
            props = self._get()
        except rospy.ServiceException as e:
            rospy.logwarn('[RealTimeFactorController] get_physics_properties failed: %s', e)
            return
        if self._orig_max_update_rate is None:
            self._orig_max_update_rate = props.max_update_rate
        new_rate = props.max_update_rate * speed_factor
        try:
            resp = self._set(time_step=props.time_step, max_update_rate=new_rate,
                             gravity=props.gravity, ode_config=props.ode_config)
        except rospy.ServiceException as e:
            rospy.logwarn('[RealTimeFactorController] set_physics_properties failed: %s', e)
            return
        if not resp.success:
            rospy.logwarn('[RealTimeFactorController] set_physics_properties rejected: %s',
                          resp.status_message)
        else:
            rospy.loginfo('[RealTimeFactorController] Requested %.1fx real-time '
                          '(max_update_rate %.0f -> %.0f Hz; actual factor achieved '
                          'depends on CPU headroom).', speed_factor,
                          props.max_update_rate, new_rate)

    def restore(self):
        if self._get is None or self._orig_max_update_rate is None:
            return
        try:
            props = self._get()
            self._set(time_step=props.time_step, max_update_rate=self._orig_max_update_rate,
                      gravity=props.gravity, ode_config=props.ode_config)
            rospy.loginfo('[RealTimeFactorController] Restored max_update_rate to %.0f Hz.',
                          self._orig_max_update_rate)
        except rospy.ServiceException as e:
            rospy.logwarn('[RealTimeFactorController] Failed to restore physics properties: %s', e)
        finally:
            self._orig_max_update_rate = None   # idempotent -- a second restore() no-ops


class SimHealthMonitor:
    """Detects two failure modes that otherwise silently corrupt a trial's
    logged data rather than cleanly failing it, for ALL THREE sweeps
    (each orchestrator AND each logger creates its OWN instance, same
    dual-instantiation pattern as GoalCompletionWatcher/ClassificationWatcher
    above -- independent subscriptions to the same topics, not a shared
    object across nodes):

    1. Gazebo physics instability ("the simulation explodes and the robot
       gets randomly launched away"): detected via (a) an implausibly large
       position jump between consecutive /slam/pose samples -- ANYmal's
       fastest configured speed anywhere in this stack is well under 1 m/s
       (probe/backup speeds, slow_approach_speed), so any inter-sample jump
       distance far beyond that is a teleport, not real locomotion; (b)
       |roll| or |pitch| exceeding a tip-over threshold -- normal walking
       and contact events stay within a few degrees, so 45 degrees is a
       comfortably conservative bound with no false positives expected from
       genuine gait/impact dynamics; (c) base height outside a sane band --
       ANYmal's nominal base height is ~0.55m AGL, so a reading far outside
       that means the robot fell through the floor or is airborne.

    2. Stall ("the robot stops and doesn't do anything"): cumulative
       displacement staying below stall_min_movement_m for a full
       stall_window_s while contact_probe_manager's own state is one where
       forward progress is actually expected (NORMAL_NAV or SLOW_APPROACH --
       NOT during CONTACT_PROBE/CLASSIFYING/BACKING_UP, where near-zero net
       motion is normal/intended). The anchor resets on any real movement,
       so a slow but genuinely-progressing robot never falsely triggers this.

    Once tripped, the reason LATCHES for the rest of the trial (call arm()
    to reset at the next trial's start) -- a momentary explosion has already
    corrupted whatever this trial measures even if Gazebo's contact solver
    happens to recover on its own afterward.

    Callers use this two ways: orchestrators poll `.anomaly()` in their
    wait loop to cut a doomed trial short instead of idling out the full
    trial_timeout; loggers read `.anomaly()` at _finish_trial() time and
    record it as a column so plotting code can exclude corrupted rows
    from aggregate stats instead of silently letting one bad trial skew a
    parameter cell's mean/std.
    """

    MOVEMENT_EXPECTED_STATES = ('NORMAL_NAV', 'SLOW_APPROACH')

    def __init__(self, jump_dist_thresh_m=1.0, tip_over_deg=45.0,
                height_min_m=0.15, height_max_m=1.5,
                stall_window_s=8.0, stall_min_movement_m=0.05):
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self._jump_thresh = jump_dist_thresh_m
        self._tip_deg = tip_over_deg
        self._h_min = height_min_m
        self._h_max = height_max_m
        self._stall_window = stall_window_s
        self._stall_min_move = stall_min_movement_m

        self._state = None
        self._reason = None          # None (healthy) or a description string, latched
        self._prev_pose = None       # (t, x, y, z)
        self._stall_anchor = None    # (t, x, y), reset whenever real movement is seen

        rospy.Subscriber('/haptiquad_nav/state', String, self._state_cb, queue_size=10)
        rospy.Subscriber('/slam/pose', PoseWithCovarianceStamped, self._pose_cb, queue_size=1)

    def arm(self):
        """Call once per trial, at trial start."""
        self._reason = None
        self._prev_pose = None
        self._stall_anchor = None

    def anomaly(self):
        """None if healthy so far this trial, else a human-readable reason
        (latched -- stays set once tripped, even if the underlying signal
        recovers)."""
        return self._reason

    def _flag(self, reason):
        if self._reason is None:   # latch to the FIRST anomaly only
            self._reason = reason
            rospy.logwarn('[SimHealthMonitor] %s', reason)

    def _state_cb(self, msg):
        self._state = msg.data
        if self._state not in self.MOVEMENT_EXPECTED_STATES:
            self._stall_anchor = None   # not tracking stall outside these states

    def _pose_cb(self, msg):
        p = msg.pose.pose.position
        t = msg.header.stamp.to_sec()
        x, y, z = p.x, p.y, p.z

        if self._prev_pose is not None:
            pt, px, py, pz = self._prev_pose
            dt = t - pt
            if dt > 1e-3:
                dist = math.sqrt((x - px)**2 + (y - py)**2 + (z - pz)**2)
                if dist > self._jump_thresh:
                    self._flag('position jumped %.2fm in %.3fs -- likely a Gazebo '
                              'physics explosion/teleport' % (dist, dt))
        self._prev_pose = (t, x, y, z)

        q = msg.pose.pose.orientation
        roll, pitch, _ = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        roll_deg, pitch_deg = math.degrees(roll), math.degrees(pitch)
        if abs(roll_deg) > self._tip_deg or abs(pitch_deg) > self._tip_deg:
            self._flag('roll/pitch (%.0f/%.0f deg) exceeds tip-over threshold '
                      '(%.0f deg)' % (roll_deg, pitch_deg, self._tip_deg))

        if not (self._h_min <= z <= self._h_max):
            self._flag('base height %.2fm outside sane range [%.2f, %.2f]m' %
                      (z, self._h_min, self._h_max))

        if self._state in self.MOVEMENT_EXPECTED_STATES:
            if self._stall_anchor is None:
                self._stall_anchor = (t, x, y)
            else:
                at, ax, ay = self._stall_anchor
                if math.hypot(x - ax, y - ay) > self._stall_min_move:
                    self._stall_anchor = (t, x, y)   # real movement seen -- reset anchor
                elif (t - at) > self._stall_window:
                    self._flag('no net movement (>%.2fm) for %.0fs while in %s -- '
                              'robot appears stalled' %
                              (self._stall_min_move, self._stall_window, self._state))


class StateWatcher:
    """Subscribes to /haptiquad_nav/state and exposes edge-detection flags
    used to bound a trial and detect the CONTACT_PROBE entry needed before
    a timeout counts as "no contact"."""

    def __init__(self):
        self.state = None
        self.saw_contact_probe = False
        rospy.Subscriber('/haptiquad_nav/state', String, self._cb, queue_size=10)

    def _cb(self, msg):
        self.state = msg.data
        if msg.data == 'CONTACT_PROBE':
            self.saw_contact_probe = True

    def reset_edges(self):
        self.saw_contact_probe = False
