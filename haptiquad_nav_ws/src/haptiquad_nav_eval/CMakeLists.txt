cmake_minimum_required(VERSION 3.0.2)
project(haptiquad_nav_eval)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  std_srvs
  geometry_msgs
  gazebo_msgs
  diagnostic_msgs
  actionlib_msgs
  haptiquad_msgs
)

# trial_core.py lives under src/haptiquad_nav_eval/ and is imported as
# haptiquad_nav_eval.trial_core -- see that file's own header comment for
# why it must NOT be listed in catkin_install_python below (that wraps
# every listed script for standalone execution, which silently breaks
# `from trial_core import X`; same pattern haptiquad_plot uses for its
# src/haptiquad_plot/ helper modules).
# Must come before generate_messages() -- catkin_python_setup.cmake enforces this.
catkin_python_setup()

add_message_files(
  FILES
  TrialStart.msg
)

generate_messages(
  DEPENDENCIES std_msgs geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs std_srvs geometry_msgs gazebo_msgs
    diagnostic_msgs actionlib_msgs haptiquad_msgs
)

catkin_install_python(PROGRAMS
  scripts/sweep_orchestrator.py
  scripts/trial_logger.py
  scripts/plot_results.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
