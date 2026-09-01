# Published by sweep_orchestrator.py right before it sends the trial's
# approach goal. This is the ONLY handoff of trial metadata to
# trial_logger.py -- the logger never talks to Gazebo or the world SDF
# itself. header.stamp doubles as t_goal_sent (the goal is sent immediately
# after this publish, in the same function call).
#
# mass / mu / geometry / size are ground truth exactly as spawned. No
# derived "is this really movable" label is computed here -- the
# contact-probe state machine's own classification (from
# /haptiquad_nav/state) IS the result under test, not something to be
# checked against an independent ground truth.

Header header
uint32 trial_id
string experiment_tag
string object_name        # spawned Gazebo model name for this trial

float64 mass               # kg
float64 mu                 # ODE friction coefficient (mu == mu2)
string geometry             # "box" | "cylinder"
float64 size_x              # box: x size (m) | cylinder: diameter (m)
float64 size_y              # box: y size (m) | cylinder: diameter (m)
float64 size_z              # height (m)
float64 spawn_x             # map frame
float64 spawn_y             # map frame

# Commanded approach geometry for this trial, relative to the robot's pose
# right after the world/robot reset (read fresh each trial, not assumed).
float64 approach_angle      # rad, map-frame heading the robot drives in on
float64 approach_offset     # m, lateral offset of the approach line from the object center
float64 robot_origin_x      # map frame, robot pose immediately after reset
float64 robot_origin_y
float64 robot_origin_yaw
