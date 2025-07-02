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
    // Initialize ROS properly to avoid the "You must call ros::init() before creating the first NodeHandle" error
    ros::init(argc, argv, "fast_lv_preprocess", ros::init_options::AnonymousName);
    ros::NodeHandle nh;
    
    // Get the config file path from command line arguments
    std::string config_file;
    if (argc > 1) {
        config_file = argv[1];
    } else {
        // Use default config file
        config_file = "/home/zzy/SensorCalibration/EasyColor/src/FastLVMapping/config/Kitti.yaml";
    }
    
    std::cout << "Using configuration file: " << config_file << std::endl;
    
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
