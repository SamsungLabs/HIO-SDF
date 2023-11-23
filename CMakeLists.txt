cmake_minimum_required(VERSION 3.0.2)
project(hio_sdf)

find_package(catkin REQUIRED COMPONENTS
  rospy
  tf2
  std_msgs
  nav_msgs
  visualization_msgs
  sensor_msgs
  geometry_msgs
)

catkin_python_setup()

catkin_package()

install(PROGRAMS
   nodes/*
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
