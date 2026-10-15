#!/usr/bin/env bash
# run_sweep.sh -- fully unattended run of one haptiquad_nav_eval sweep.
#
# Run on the HOST from anywhere:
#   haptiquad_nav_ws/src/haptiquad_nav_eval/tools/run_sweep.sh <mass|tuning|timing> [tag]
#   haptiquad_nav_ws/src/haptiquad_nav_eval/tools/run_sweep.sh status [tag...]
#       trials done/remaining, s/trial and ETA (all runner-managed tags if none
#       given); the same line is kept in <tag>/STATUS while a sweep runs.
#
# No GUI is opened: the ANYmal rviz, rqt_gui and gazebo_gui are all disabled
# (sim_eval.launch rviz:=false) and navigation.launch's rviz is off.
#
# Brings up the container if needed, builds, then per attempt starts a
# private roscore + sim_eval.launch + navigation.launch + the eval launch,
# watches progress, and tears everything down. A stalled or crashed stack is
# killed and relaunched; the orchestrators resume from progress.csv. Finished
# when the orchestrator writes <tag>/DONE.
#
# Isolation: ROS master on 11411 and Gazebo master on 11445 (not the
# defaults), so nothing else using ROS/Gazebo on this host is touched.
# Because the container runs with pid: host, cleanup never kills by process
# name -- only processes carrying this run's HAPTIQUAD_RUN_ID in their
# environment.
#
# Env overrides: MAX_ATTEMPTS (default 20), STALL_MIN (minutes without a new
# completed trial before restarting, default 10), SKIP_BUILD=1.

set -u

WS_CONTAINER=/home/anydev/anymal_ws
CONTAINER=haptiquad_nav_sim
ROS_PORT=11411
GZ_PORT=11445

# ─────────────────────────────── inner (container) ──────────────────────────
if [[ "${1:-}" == "--inner" ]]; then
  launch_file=$2; tag=$3; attempt=$4; stall_min=$5
  tag_dir=$WS_CONTAINER/src/haptiquad_nav_eval/.results/$tag
  log_dir=$tag_dir/runner_logs/attempt_$attempt
  mkdir -p "$log_dir"

  source /opt/ros/noetic/setup.bash
  source $WS_CONTAINER/devel/setup.bash
  export ROS_MASTER_URI=http://localhost:$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GZ_PORT
  export HAPTIQUAD_RUN_ID="haptiquad_sweep_${tag}_$$"

  log() { echo "[$(date '+%F %T')] $*" | tee -a "$log_dir/runner.log"; }

  pids=()
  start() {  # start <name> <cmd...> -- own session so the whole tree can be signalled
    local name=$1; shift
    setsid "$@" >"$log_dir/$name.log" 2>&1 &
    pids+=($!)
    log "started $name (pid $!)"
  }

  cleanup() {
    log "cleanup"
    for p in "${pids[@]}"; do kill -INT -- "-$p" 2>/dev/null; done
    sleep 10
    for p in "${pids[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done
    # Anything that escaped its session (daemonised stack_launcher children,
    # gzserver): match on this run's env token only.
    for e in /proc/[0-9]*/environ; do
      if grep -qz "^HAPTIQUAD_RUN_ID=$HAPTIQUAD_RUN_ID\$" "$e" 2>/dev/null; then
        p=${e#/proc/}; p=${p%/environ}
        [[ $p != $$ ]] && kill -KILL "$p" 2>/dev/null
      fi
    done
  }
  trap cleanup EXIT
  trap 'exit 4' INT TERM

  alive() { kill -0 "$1" 2>/dev/null; }
  wait_for() {  # wait_for <timeout_s> <description> <cmd...>
    local t=$1 what=$2; shift 2
    for ((i = 0; i < t; i += 2)); do
      "$@" >/dev/null 2>&1 && { log "ready: $what"; return 0; }
      sleep 2
    done
    log "TIMEOUT waiting for $what"; return 1
  }
  init_done() { ! rosnode list 2>/dev/null | grep -q '^/anymal_motion_init$'; }

  start roscore roscore -p $ROS_PORT
  wait_for 60 roscore rosparam list || exit 3

  start sim roslaunch haptiquad_anymal_bringup sim_eval.launch rviz:=false
  sim_pid=${pids[-1]}
  wait_for 240 /gazebo/reset_world rosservice info /gazebo/reset_world \
    && wait_for 240 /state_estimator/reset rosservice info /state_estimator/reset || exit 3

  start nav roslaunch haptiquad_nav navigation.launch rviz:=false init_robot:=true
  nav_pid=${pids[-1]}
  wait_for 120 contact_probe_manager rosservice info /contact_probe_manager/reset \
    && wait_for 120 move_base rosservice info /move_base/clear_costmaps || exit 3
  sleep 5   # let anymal_motion_init register before polling for its exit
  wait_for 120 "e-stop clear + walk mode" init_done || exit 3
  sleep 5

  start eval roslaunch haptiquad_nav_eval "$launch_file" tag:="$tag"
  eval_pid=${pids[-1]}

  sample() { python3 $WS_CONTAINER/src/haptiquad_nav_eval/tools/sweep_status.py --sample "$tag_dir"; }

  progress=$tag_dir/csv/progress.csv
  last_change=$(date +%s)
  last_mtime=0
  while true; do
    sleep 30
    sample
    if [[ -f $tag_dir/DONE ]]; then log "sweep finished"; exit 0; fi
    for n in sim_pid nav_pid eval_pid; do
      if ! alive "${!n}"; then log "${n%_pid} launch exited unexpectedly"; exit 3; fi
    done
    m=$(stat -c %Y "$progress" 2>/dev/null || echo 0)
    if [[ $m != "$last_mtime" ]]; then last_mtime=$m; last_change=$(date +%s); fi
    if (( $(date +%s) - last_change > stall_min * 60 )); then
      log "no completed trial for ${stall_min} min -- restarting stack"; exit 3
    fi
  done
fi

# ─────────────────────────────── outer (host) ───────────────────────────────
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
results_host=$script_dir/../.results

if [[ "${1:-}" == "status" ]]; then
  shift
  dirs=()
  if (( $# )); then
    for t in "$@"; do dirs+=("$results_host/$t"); done
  else
    for d in "$results_host"/*/; do [[ -d $d/runner_logs ]] && dirs+=("${d%/}"); done
  fi
  (( ${#dirs[@]} )) || { echo "no runner-managed sweeps found"; exit 0; }
  exec python3 "$script_dir/sweep_status.py" "${dirs[@]}"
fi

case "${1:-}" in
  mass)   launch_file=eval_sweep.launch;        default_tag=mass_mu_final ;;
  tuning) launch_file=eval_tuning_sweep.launch; default_tag=tuning_sweep_02 ;;
  timing) launch_file=eval_timing_sweep.launch; default_tag=timing_final ;;
  *) echo "usage: $0 <mass|tuning|timing> [tag]  |  $0 status [tag...]" >&2; exit 2 ;;
esac
tag=${2:-$default_tag}
max_attempts=${MAX_ATTEMPTS:-20}
stall_min=${STALL_MIN:-10}

repo=$(cd "$script_dir/../../../.." && pwd)
tag_dir_host=$results_host/$tag
inner=$WS_CONTAINER/src/haptiquad_nav_eval/tools/run_sweep.sh

hlog() { echo "[$(date '+%F %T')] $*"; }

if [[ -f $tag_dir_host/DONE ]]; then hlog "$tag already finished"; exit 0; fi
if [[ -n "$(ls -A "$tag_dir_host/plots" 2>/dev/null)" ]]; then
  hlog "$tag has plots but no DONE marker (finished by an older harness) -- pick another tag"; exit 2
fi

if ! docker ps --format '{{.Names}}' | grep -qx $CONTAINER; then
  hlog "starting container"
  docker compose -f "$repo/compose.yaml" up -d || exit 1
fi

mkdir -p "$tag_dir_host/runner_logs"

# docker exec does not forward Ctrl-C; pass it on so the inner side cleans up.
# Matching on the full "--inner <launch> <tag>" argv is specific to this run.
trap 'hlog "stopping"; docker exec $CONTAINER pkill -INT -f "run_sweep.sh --inner $launch_file $tag " ; exit 4' INT TERM

if [[ -z "${SKIP_BUILD:-}" ]]; then
  hlog "building"
  docker exec $CONTAINER bash -c "source /opt/ros/noetic/setup.bash && source $WS_CONTAINER/devel/setup.bash \
    && cd $WS_CONTAINER && catkin_make --pkg haptiquad_ros haptiquad_nav haptiquad_nav_eval haptiquad_anymal_bringup" \
    > "$tag_dir_host/runner_logs/build.log" 2>&1 \
    || { hlog "build failed, see $tag_dir_host/runner_logs/build.log"; exit 1; }
fi

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  hlog "$tag: attempt $attempt/$max_attempts"
  docker exec $CONTAINER bash "$inner" --inner "$launch_file" "$tag" "$attempt" "$stall_min"
  rc=$?
  if [[ -f $tag_dir_host/DONE ]]; then hlog "$tag: done ($(cat "$tag_dir_host/DONE" | tr -d '\n') trials)"; exit 0; fi
  [[ $rc == 4 ]] && { hlog "interrupted"; exit 4; }
  hlog "$tag: attempt $attempt ended (rc=$rc) without DONE -- relaunching in 20s"
  sleep 20
done
hlog "$tag: gave up after $max_attempts attempts"; exit 1
