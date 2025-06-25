cmake_minimum_required(VERSION 3.0.2)
project(fast_lv_mapping)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译优化标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")

# 添加OpenMP支持 - 改进检测和标志设置
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DOPENMP_FOUND=1)
    message(STATUS "Building with OpenMP support")
else()
    add_definitions(-DOPENMP_FOUND=0)
    message(STATUS "OpenMP not found. Parallel processing disabled.")
endif()

# 找到必要的包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
  tf2_ros
  tf2
  visualization_msgs
  image_transport
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

# Catkin包设置
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES fast_lv_mapping
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs nav_msgs geometry_msgs cv_bridge pcl_ros pcl_conversions tf2_ros tf2 visualization_msgs image_transport
  DEPENDS OpenCV PCL EIGEN3
)

# 设置包含目录
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# 从源文件创建库
add_library(fast_lv_mapping
  src/calib.cpp
  src/config.cpp
  src/ImageProcess/imageprocess.cpp
  # 添加其他源文件（如需）
)

# 将库与依赖项链接
target_link_libraries(fast_lv_mapping
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fast_lv_mapping ${OpenMP_CXX_LIBRARIES})
endif()

# 创建ROS节点可执行文件
add_executable(fast_lv_mapping_node src/ros_node.cpp)
target_link_libraries(fast_lv_mapping_node
  fast_lv_mapping
  ${catkin_LIBRARIES}
)

# 创建常规可执行文件（非ROS）
add_executable(fast_lv_mapping_run src/run.cpp)
target_link_libraries(fast_lv_mapping_run
  fast_lv_mapping
)

# 创建预处理可执行文件（非ROS）
add_executable(fast_lv_mapping_preprocess src/preprocess.cpp)
target_link_libraries(fast_lv_mapping_preprocess
  fast_lv_mapping
)