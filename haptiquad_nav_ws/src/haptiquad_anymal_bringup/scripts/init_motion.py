#!/usr/bin/env python3
"""
One-shot node: clears the ANYmal emergency stop, then switches the
motion controller to trekker/walk mode ready for navigation.

Run via:  rosrun haptiquad_anymal_bringup init_motion.py
Or included in a launch file — exits automatically when done.
"""
import subprocess
import rospy
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
    else:
        rospy.loginfo('[AnymalInit] Motion state initialized — robot ready for navigation.')

    rospy.signal_shutdown('init complete')


if __name__ == '__main__':
    main()
