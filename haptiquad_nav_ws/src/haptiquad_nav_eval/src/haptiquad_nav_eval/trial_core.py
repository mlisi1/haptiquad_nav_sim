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

import rospy
import tf.transformations
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
from std_msgs.msg import String
from std_srvs.srv import Empty as EmptySrv
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetWorldProperties


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
