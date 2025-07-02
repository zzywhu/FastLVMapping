#include "calib.h"
#include "config.h"
#include <iostream>
#include <string>
#include <ros/ros.h>

/**
 * @brief Main function to run preprocessing only
 * This creates index maps and projections needed for calibration
 */
int main(int argc, char** argv)
{
    // 初始化ROS
    ros::init(argc, argv, "fast_lv_preprocess", ros::init_options::AnonymousName);
    ros::NodeHandle nh("~"); // 使用私有命名空间，这很重要！
    
    // 获取配置文件路径
    std::string config_file;
    if (!nh.getParam("config_file", config_file)) {
        // 如果参数未设置，使用默认值
        config_file = "/home/zzy/SensorCalibration/EasyColor/src/FastLVMapping/config/default_config.yaml";
        ROS_WARN("No config_file parameter provided, using default: %s", config_file.c_str());
    } else {
        ROS_INFO("Using configuration file: %s", config_file.c_str());
    }
    
    // Create processor without a ROS node handle to avoid publishing
    lvmapping::CalibProcessor processor;
    
    // Initialize with config file
    if (!processor.initialize(config_file)) {
        std::cerr << "Failed to initialize processor with config: " << config_file << std::endl;
        return 1;
    }
    
    // Run preprocessing function
    std::cout << "Starting preprocessing..." << std::endl;
    if (!processor.runPreprocessing()) {
        std::cerr << "Preprocessing failed!" << std::endl;
        return 1;
    }
    
    std::cout << "Preprocessing completed successfully!" << std::endl;
    return 0;
}
