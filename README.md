# HaptiQuad Navigation Simulator

Tactile-aware navigation for ANYmal C. The robot navigates through an all-free map and classifies physical contacts as movable (push through) or solid (replan) using only HaptiQuad's base wrench estimate — no camera-based avoidance.

---

## Prerequisites

- Docker running with the `haptiquad_nav_sim` container
- ANYmal simulation and HaptiQuad stack already started inside the container

---

## Build (first time or after code changes)

```bash
docker compose -f compose.yaml exec haptiquad_nav_sim bash
source /opt/ros/noetic/setup.bash
source /home/anydev/anymal_ws/devel/setup.bash
cd /home/anydev/anymal_ws
catkin_make --pkg haptiquad_nav haptiquad_anymal_bringup
```

---

## Launch

```bash
roslaunch haptiquad_nav navigation.launch
```

This automatically:
1. Clears the ANYmal emergency stop
2. Switches the motion controller to trekker/walk mode
3. Starts move_base (global + local costmaps, custom `HaptiQuadNavPlanner` local planner), the odometry bridge, wrench bridge, contact probe manager, twist stamper, RViz goal relay, and the debug perception costmap

RViz is **off by default**. Open it with:

```bash
roslaunch haptiquad_nav navigation.launch rviz:=true
```

If the robot is already in walk mode (e.g. relaunching nav only):

```bash
roslaunch haptiquad_nav navigation.launch init_robot:=false
```

---

## Send a Navigation Goal

Use the **2D Nav Goal** button in RViz. Goals are relayed from `/goal` (ANYmal RViz default) to `/move_base_simple/goal` automatically.

Or from the command line:

```bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: 14.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'
```

---

## Key Topics to Monitor

| Topic | What it tells you |
|-------|------------------|
| `/anymal/base_wrench` | Contact force on base (world frame) — the contact-detection signal |
| `/haptiquad_nav/speed_limit` | Current speed cap published by the contact probe state machine |
| `/haptiquad_nav/solid_obstacles` | Accumulated confirmed-solid obstacle points (fed into the global costmap) |
| `/haptiquad_nav/movable_obstacles` | Tracked centers of confirmed-movable obstacles (visualization only, not in any costmap) |
| `/move_base/status` | Active navigation goal status |
| `/path_planning_and_following/twist` | Gated velocity output (after the contact probe state machine and twist_mux) |

---

## Test Worlds

Two worlds are available under `haptiquad_anymal_bringup/worlds/`. The active world is set by **two** references in `haptiquad_anymal_bringup/config/default.yaml` (the Gazebo server args and the `rviz_world_loader` config) — both must point to the same file.

### `mixed_obstacles_test.world` (current default)

9 objects at varied x/y positions (x ≈ 2.5–12.0 m, y ≈ −0.8 to +0.9 m), mix of boxes and cylinders, all 1.0 m tall. Send a goal at `(14.0, 0.0)` in map frame.

| Object | Shape | Mass | μ | Color | Expected |
|--------|-------|------|---|-------|---------|
| solid_box_1..4 | 0.5×0.5×1.0 m box | 200 kg | 0.8 | Red | SOLID |
| solid_cyl_1..2 | r=0.25, h=1.0 m cylinder | 200 kg | 0.8 | Red | SOLID |
| movable_box_1..2 | 0.5×0.5×1.0 m box | 5 kg | 0.05 | Green | MOVABLE |
| movable_cyl_1 | r=0.25, h=1.0 m cylinder | 5 kg | 0.05 | Green | MOVABLE |

### `contact_classification_test.world` (reference / regression)

Five cubes in a row along +X (robot starts at origin), 0.5×0.5×1.0 m each, 1.5 m center-to-center. Send a goal at `(10.0, 0.0)` in map frame.

| Name | x (m) | Mass | μ | Color | Expected |
|------|--------|------|---|-------|---------|
| cube_light_1 | 2.0 | 5 kg | 0.1 | Green | MOVABLE |
| cube_heavy_1 | 3.5 | 200 kg | 0.8 | Red | SOLID → replan |
| cube_light_2 | 5.0 | 5 kg | 0.1 | Green | MOVABLE |
| cube_heavy_2 | 6.5 | 200 kg | 0.8 | Red | SOLID → replan |
| cube_heavy_3 | 8.0 | 200 kg | 0.8 | Red | SOLID → replan |

---

## Architecture Summary

See `SYSTEM.md` for the full node/topic graph and `CLAUDE.md` for implementation details and pitfalls.

See `EXPERIMENTS.md` for the three unattended evaluation harnesses in `haptiquad_nav_eval` — mass/friction classification, lateral-length task-time comparison, and contact-probe parameter tuning — their data logging, config-driven plot generation (per-plot YAML config + self-contained `plot_scripts/` bundles), and the ablation study between force- and displacement-based classification.
