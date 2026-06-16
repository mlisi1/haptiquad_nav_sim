cmake_minimum_required(VERSION 2.8.3)
project(haptiquad_anymal_bringup)

find_package(catkin REQUIRED COMPONENTS rospy std_srvs)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/init_motion.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
