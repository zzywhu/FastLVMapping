#include "calib.h"
#include "config.h"
#include <iostream>
#include <string>
#include <cstdlib> // For system() function

/**
 * @brief Check if a command is available in system PATH
 * @param command Command to check
 * @return True if command is available
 */
bool isCommandAvailable(const std::string &command) {
    std::string test_cmd = "which " + command + " > /dev/null 2>&1";
    return system(test_cmd.c_str()) == 0;
}

/**
 * @brief Suggest visualization tools to the user
 * @param pcd_path Path to the colored point cloud
 */
void suggestVisualizationTools(const std::string &pcd_path) {
    std::cout << "\n=== Point Cloud Visualization Options ===" << std::endl;

    if (isCommandAvailable("pcl_viewer")) {
        std::cout << "Option 1: Use PCL Viewer: pcl_viewer " << pcd_path << std::endl;
    } else {
        std::cout << "PCL Viewer not found. You can install it with: sudo apt install pcl-tools" << std::endl;
    }

    if (isCommandAvailable("cloudcompare")) {
        std::cout << "Option 2: Use CloudCompare: cloudcompare " << pcd_path << std::endl;
    } else {
        std::cout << "CloudCompare not found. You can install it with: sudo apt install cloudcompare" << std::endl;
    }

    if (isCommandAvailable("meshlab")) {
        std::cout << "Option 3: Use MeshLab: meshlab " << pcd_path << std::endl;
    } else {
        std::cout << "MeshLab not found. You can install it with: sudo apt install meshlab" << std::endl;
    }

    std::cout << "You can open the colored point cloud with any PCD viewer application.\n"
              << std::endl;
}

/**
 * @brief Main function to run the calibration and colorization process
 */
int main(int argc, char **argv) {
    // Default configuration file path - fixed to correct path
    std::string config_file = "/home/zzy/SensorCalibration/EasyColor/src/FastLVMapping/config/default_config.yaml";

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

    // Check for visualization tools
    std::string viewer_command = "pcl_viewer"; // Default viewer

    if (!isCommandAvailable(viewer_command)) {
        std::cout << "Warning: pcl_viewer not found. Checking for alternatives..." << std::endl;

        // Check alternatives
        if (isCommandAvailable("cloudcompare")) {
            viewer_command = "cloudcompare";
        } else if (isCommandAvailable("meshlab")) {
            viewer_command = "meshlab";
        } else {
            std::cout << "No suitable point cloud viewer found. You'll need to install one." << std::endl;
            std::cout << "Processing will continue but without real-time visualization." << std::endl;
            viewer_command = "";
        }
    }
    // Set visualization tool
    calib.setVisualizationTool(viewer_command);

    // Run the main calibration and colorization process
    bool success = calib.run();

    // Calculate path of final point cloud for visualization suggestion
    std::string output_path = lvmapping::Config::getInstance().processingParams().output_path;
    std::string colormap_dir = output_path + "/colormap";
    std::string colored_pcd_path = colormap_dir + "/colored_pointcloud_live.pcd";

    // Print results
    if (success) {
        std::cout << "\n===========================================";
        std::cout << "\nSuccessfully processed all images and created projections." << std::endl;
        std::cout << "Results saved to: " << output_path << std::endl;
        std::cout << "Generated files include:" << std::endl;
        std::cout << " - Colored point cloud: " << colored_pcd_path << std::endl;
        std::cout << " - Camera trajectory visualization: " << output_path << "/trajectory/camera_trajectory_with_model.pcd" << std::endl;
        std::cout << " - Optimized camera poses: " << output_path << "/trajectory/optimized_camera_poses.txt" << std::endl;
        std::cout << " - Optimized extrinsics: " << output_path << "/calibration/optimized_extrinsics.txt" << std::endl;

        // Suggest visualization tools
        suggestVisualizationTools(colored_pcd_path);
    } else {
        std::cerr << "Error during image processing and projection." << std::endl;
        return -1;
    }

    return 0;
}
