cmake_minimum_required(VERSION 3.10)
project(FastLVMapping)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(cv_bridge REQUIRED)
# Find ROS packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
)

# Include directories
include_directories(
    include
    include/ImageProcess
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${catkin_INCLUDE_DIRS}
)
SET(cv_bridge_LIBRARIES "/opt/ros/noetic/lib/libcv_bridge.so")
# Link directories
link_directories(${PCL_LIBRARY_DIRS})

# Add definitions from PCL
add_definitions(${PCL_DEFINITIONS})

# Set source files
set(SOURCES
    src/calib.cpp
    src/ImageProcess/imageprocess.cpp
    # Add other source files here
)


# Add another executable for calibration if needed

add_library(common_lib
    src/calib.cpp
    src/config.cpp  # Added config.cpp
    # ... other common source files
)

# If you have a preprocessing target, also add it there
if(TARGET preprocessing)
    target_sources(preprocessing PRIVATE src/config.cpp)
endif()

add_executable(calibration src/run.cpp ${SOURCES})
target_link_libraries(calibration
    ${OpenCV_LIBS}
    common_lib
    ${PCL_LIBRARIES}
    stdc++fs
)

add_executable(preprocess src/preprocess.cpp ${SOURCES})
target_link_libraries(preprocess
    ${OpenCV_LIBS}
    common_lib
    ${PCL_LIBRARIES}
    stdc++fs
)