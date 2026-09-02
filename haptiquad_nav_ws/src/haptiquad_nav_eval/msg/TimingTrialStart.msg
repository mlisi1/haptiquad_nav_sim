# Published by timing_sweep_orchestrator.py right before it sends the
# trial's approach goal -- the only handoff of trial metadata to
# timing_trial_logger.py. Deliberately a SEPARATE message from TrialStart.msg
# (used by the mass/mu sweep's sweep_orchestrator.py/trial_logger.py) so the
# two experiments never share a schema and can't affect each other.
#
# header.stamp doubles as t_goal_sent (the goal is sent immediately after
# this publish, in the same function call).

Header header
uint32 trial_id
string experiment_tag

# "classical"  -- contact-probing bypassed (~set_probing_enabled false);
#                 the obstacle is pre-known/never touched, pure move_base
#                 circumvent-navigation baseline.
# "movable"    -- probing enabled; object's mass/mu chosen to be pushable.
# "solid"      -- probing enabled; object's mass/mu chosen to be immovable.
string trial_mode

string object_name        # spawned Gazebo model name for this trial

float64 mass               # kg
float64 mu                 # ODE friction coefficient (mu == mu2)
string geometry             # "box" | "cylinder"
float64 size_x              # box: x size (m, depth along approach) | cylinder: diameter (m)
float64 size_y              # box: y size (m, lateral length -- the swept variable) | cylinder: diameter (m)
float64 size_z              # height (m)
float64 spawn_x             # map frame
float64 spawn_y             # map frame

float64 goal_x              # map frame -- commanded approach-goal position
float64 goal_y

float64 approach_angle      # rad, map-frame heading the robot drives in on (fixed 0 for this experiment)
float64 robot_origin_x      # map frame, robot pose immediately after reset
float64 robot_origin_y
float64 robot_origin_yaw
