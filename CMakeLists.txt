cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(correspondence_grouping)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#add_executable (correspondence_grouping src/correspondence_grouping.cpp)
#target_link_libraries (correspondence_grouping ${PCL_LIBRARIES})

add_executable (global_feature_classification src/global_feature_classification.cpp src/global_features_model.hpp)
target_link_libraries (global_feature_classification ${PCL_LIBRARIES})

add_executable (pcl_visualizer_demo src/pcl_visualizer_demo.cpp)
target_link_libraries (pcl_visualizer_demo ${PCL_LIBRARIES})
