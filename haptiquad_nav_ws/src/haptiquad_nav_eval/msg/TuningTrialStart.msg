# Published by tuning_sweep_orchestrator.py right before it sends the
# trial's approach goal -- the only handoff of trial metadata to
# tuning_trial_logger.py. Deliberately a SEPARATE message from TrialStart.msg
# (mass/mu sweep) and TimingTrialStart.msg (lateral-length sweep) so all
# three experiments never share a schema and can't affect each other.
#
# header.stamp doubles as t_goal_sent (the goal is sent immediately after
# this publish, in the same function call).

Header header
uint32 trial_id
string experiment_tag

# "movable" | "solid" -- which ground-truth arm this trial is. Mass/mu are
# fixed per arm (see tuning_sweep.yaml), NOT swept here -- only contact-probe
# TIMING parameters are.
string trial_mode

string object_name        # spawned Gazebo model name for this trial

float64 mass               # kg
float64 mu                 # ODE friction coefficient (mu == mu2)
string geometry             # "box" | "cylinder"
float64 size_x              # m (fixed, not swept, this experiment)
float64 size_y              # m
float64 size_z              # m
float64 spawn_x             # map frame
float64 spawn_y             # map frame

float64 goal_x              # map frame -- commanded approach-goal position
float64 goal_y

float64 approach_angle      # rad, map-frame heading the robot drives in on (fixed 0)
float64 robot_origin_x      # map frame, robot pose immediately after reset
float64 robot_origin_y
float64 robot_origin_yaw

# Index of this trial's parameter combination in the full-factorial grid
# (shared by both arms and all repeats of that combination).
uint32 combo_id

# Value of every tunable parameter used for this trial. Logged explicitly so
# summary.csv is self-describing per row -- tuning_plotting.py's "vs param"
# curves average over all other parameters (main effects of a balanced
# factorial design).
float64 probe_speed
float64 probe_distance
float64 contact_hold_time
float64 classify_time
float64 classification_force_settle_time
int32   wrench_filter_window
