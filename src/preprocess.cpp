#include "calib.h"
#include "config.h"
#include <iostream>
#include <string>

/**
 * @brief Main function to run preprocessing only
 * This creates index maps and projections needed for calibration
 */
int main(int argc, char** argv)
{
    // Default configuration file path
    std::string config_file = "/home/zzy/SensorCalibration/FastLVMapping/config/default_config.yaml";
    
    // Command line argument can override config path
    if (argc > 1) {
        config_file = argv[1];
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
    } else {
        std::cerr << "Error during preprocessing." << std::endl;
        return -1;
    }

    return 0;
}
