#!/usr/bin/env python3
"""
One-shot node: clears the ANYmal emergency stop, then switches the
motion controller to trekker/walk mode ready for navigation.

Run via:  rosrun haptiquad_anymal_bringup init_motion.py
Or included in a launch file — exits automatically when done.

Sets ~done_param (default /haptiquad_nav/motion_initialized) to True on
success; haptiquad_nav's nav_supervisor waits for it before starting the
navigation core. Left False on any failure, so navigation never starts.
"""
import subprocess
import rospy
import yaml
from std_srvs.srv import Empty

# YAML representation of the ExecuteMotionActionGoal for trekker walk mode.
# Passed as a string to `rostopic pub -1` to avoid importing the proprietary
# anymal_motion_control_msgs package at the Python level.
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


def main():
    rospy.init_node('anymal_motion_init', anonymous=False)
    done_param = rospy.get_param('~done_param', '/haptiquad_nav/motion_initialized')
    rospy.set_param(done_param, False)

    # ── Step 1: clear emergency stop ─────────────────────────────────────────
    rospy.loginfo('[AnymalInit] Waiting for /motion_control_manager/clear_emergency_stop ...')
    try:
        rospy.wait_for_service('/motion_control_manager/clear_emergency_stop', timeout=30.0)
    except rospy.ROSException:
        rospy.logerr('[AnymalInit] Service not available after 30s — is the simulator running?')
        return

    rospy.ServiceProxy('/motion_control_manager/clear_emergency_stop', Empty)()
    rospy.loginfo('[AnymalInit] Emergency stop cleared.')
    rospy.sleep(1.0)

    # ── Step 2: switch to trekker walk mode ───────────────────────────────────
    # Listen for the action result first, so "walk mode reached" means the
    # motion control manager reported SUCCEEDED, not just that a goal was sent.
    result_timeout = rospy.get_param('~result_timeout', 30.0)
    listener = subprocess.Popen(
        ['rostopic', 'echo', '-n', '1', '/motion_control_manager/execute_motion/result'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rospy.sleep(1.0)
    rospy.loginfo('[AnymalInit] Requesting trekker/walk motion state ...')
    result = subprocess.run(
        [
            'rostopic', 'pub', '-1',
            '/motion_control_manager/execute_motion/goal',
            'anymal_motion_control_msgs/ExecuteMotionActionGoal',
            _TREKKER_WALK_GOAL,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        rospy.logerr('[AnymalInit] rostopic pub failed:\n%s', result.stderr)
        listener.kill()
        rospy.signal_shutdown('init failed')
        return

    try:
        out, _ = listener.communicate(timeout=result_timeout)
        status = yaml.safe_load(out.split('---')[0])['status']['status']
    except (subprocess.TimeoutExpired, yaml.YAMLError, KeyError, TypeError) as e:
        listener.kill()
        rospy.logerr('[AnymalInit] No valid execute_motion result within %.0fs (%s) -- '
                     'navigation will NOT start.', result_timeout, type(e).__name__)
        rospy.signal_shutdown('init failed')
        return
    if status != 3:   # actionlib_msgs/GoalStatus.SUCCEEDED
        rospy.logerr('[AnymalInit] execute_motion finished with status %s, not SUCCEEDED -- '
                     'navigation will NOT start.', status)
    else:
        rospy.loginfo('[AnymalInit] Motion state initialized — robot ready for navigation.')
        rospy.set_param(done_param, True)

    rospy.signal_shutdown('init complete')


if __name__ == '__main__':
    main()
