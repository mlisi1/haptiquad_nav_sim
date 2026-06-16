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
catkin_make --pkg haptiquad_anymal_bringup haptiquad_nav
```

---

## Launch

```bash
roslaunch haptiquad_nav navigation.launch
```

This automatically:
1. Clears the ANYmal emergency stop
2. Switches the motion controller to trekker/walk mode
3. Starts move_base (global + local costmaps), odometry bridge, wrench bridge, contact probe manager, and the RViz goal relay
4. Opens RViz with a pre-configured display layout (costmaps, plans, obstacles). Disable with `rviz:=false`

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
  '{header: {frame_id: map}, pose: {position: {x: 2.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'
```

---

## Key Topics to Monitor

| Topic | What it tells you |
|-------|------------------|
| `/anymal/base_wrench` | Contact force on base (world frame) |
| `/haptiquad_nav/solid_obstacles` | Accumulated confirmed-solid obstacle points |
| `/move_base/status` | Active navigation goal status |
| `/path_planning_and_following/twist` | Gated velocity output (after contact probe state machine) |

---

## Test World

`haptiquad_anymal_bringup/worlds/contact_classification_test.world` — load this instead of `empty.world` to test the full classification pipeline.

Five cubes in a row along +X (robot starts at origin):

| Name | x (m) | Mass | Friction | Color | Expected result |
|------|--------|------|----------|-------|----------------|
| cube_light_1 | 2.0 | 5 kg | 0.1 | Green | MOVABLE |
| cube_heavy_1 | 3.5 | 200 kg | 0.8 | Red | SOLID → replan |
| cube_light_2 | 5.0 | 5 kg | 0.1 | Green | MOVABLE |
| cube_heavy_2 | 6.5 | 200 kg | 0.8 | Red | SOLID → replan |
| cube_heavy_3 | 8.0 | 200 kg | 0.8 | Red | SOLID → replan |

Send a goal at `(10.0, 0.0)` in map frame to drive through all cubes.

---

## Architecture Summary

See `SYSTEM.md` for the full node/topic graph and `CLAUDE.md` for implementation details and pitfalls.
