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
    // Initialize ROS (even if we don't use ROS features)
    ros::init(argc, argv, "fast_lv_preprocess", ros::init_options::AnonymousName);
    
    // Get the config file path from the command line or from ROS parameter
    std::string config_file;
    if (argc > 1) {
        config_file = argv[1];
    } else {
        ros::NodeHandle pnh("~");
        if (!pnh.getParam("config_file", config_file)) {
            // Default config file if not specified
            config_file = "../config/default_config.yaml";
            ROS_WARN("No config file specified, using default: %s", config_file.c_str());
        }
    }

    std::cout << "Using configuration file: " << config_file << std::endl;
    
    // Create a CalibProcessor instance
    lvmapping::CalibProcessor calib;
    
    // Initialize calibration with config file
    if (!calib.initialize(config_file)) {
        std::cerr << "Failed to initialize calibration with config file: " << config_file << std::endl;
        return -1;
    }
    
    std::cout << "Starting preprocessing..." << std::endl;
    
    // Run preprocessing only
    bool success = calib.runPreprocessing();
    
    if (success) {
        std::cout << "\n===========================================";
        std::cout << "\nPreprocessing completed successfully." << std::endl;
        std::cout << "Now you can run the main calibration process." << std::endl;
        std::cout << "After calibration, additional outputs will include:" << std::endl;
        std::cout << " - Optimized camera poses" << std::endl;
        std::cout << " - 3D visualization of camera trajectory with camera models" << std::endl;
        std::cout << " - Optimized extrinsics parameters for each timestamp" << std::endl;
    } else {
        std::cerr << "Error during preprocessing." << std::endl;
        return -1;
    }

    return 0;
}
