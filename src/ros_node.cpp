#include "calib.h"
#include "config.h"
#include <ros/ros.h>
#include <csignal>
#include <memory>
#include <iostream>

// Global variables for signal handling
bool g_request_shutdown = false;
std::unique_ptr<lvmapping::CalibProcessor> g_calib_processor;

/**
 * @brief Signal handler for graceful shutdown with Ctrl+C
 * @param sig Signal number
 */
void signalHandler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        std::cout << "\n\n*** Shutting down FastLVMapping... ***" << std::endl;
        g_request_shutdown = true;

        // 先清理自定义资源
        if (g_calib_processor) {
            std::cout << "Cleaning up calibration processor..." << std::endl;
            // 执行清理但不强制退出
            try {
                // 让处理器自然结束当前操作
                g_calib_processor.reset(); // 释放处理器资源
            } catch (const std::exception &e) {
                ROS_WARN("Exception during cleanup: %s", e.what());
            }
        }

        // 最后请求ROS shutdown
        if (ros::ok()) {
            ros::requestShutdown();
        }

        std::cout << "Shutdown complete." << std::endl;
    }
}

/**
 * @brief Main function to run the calibration and colorization process as a ROS node
 */
int main(int argc, char **argv) {
    // Initialize ROS with custom signal handling
    ros::init(argc, argv, "fast_lv_mapping", ros::init_options::NoSigintHandler);

    // Create a node handle
    ros::NodeHandle nh("~");

    // Register custom signal handler for both SIGINT and SIGTERM
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Default configuration file path
    std::string config_file;
    if (!nh.getParam("config_file", config_file)) {
        config_file = "/home/zzy/SensorCalibration/EasyColor/src/FastLVMapping/config/default_config.yaml";
        ROS_WARN("No config_file parameter provided, using default: %s", config_file.c_str());
    } else {
        ROS_INFO("Using configuration file: %s", config_file.c_str());
    }

    // Create calibration processor with ROS node handle
    g_calib_processor = std::make_unique<lvmapping::CalibProcessor>(&nh);

    // Initialize calibration processor with config file
    if (!g_calib_processor->initialize(config_file)) {
        ROS_ERROR("Failed to initialize calibration with config file: %s", config_file.c_str());
        return -1;
    }

    // Validate required paths from loaded config
    const lvmapping::Config &config = lvmapping::Config::getInstance();
    const auto &params = config.processingParams();

    bool paths_valid = true;

    if (params.img_path.empty()) {
        ROS_ERROR("Image path is empty. Check your config file.");
        paths_valid = false;
    } else {
        ROS_INFO("Image path: %s", params.img_path.c_str());
    }

    if (params.traj_path.empty()) {
        ROS_ERROR("Trajectory path is empty. Check your config file.");
        paths_valid = false;
    } else {
        ROS_INFO("Trajectory path: %s", params.traj_path.c_str());
    }

    if (params.pcd_path.empty()) {
        ROS_ERROR("Point cloud path is empty. Check your config file.");
        paths_valid = false;
    } else {
        ROS_INFO("Point cloud path: %s", params.pcd_path.c_str());
    }

    if (params.output_path.empty()) {
        ROS_ERROR("Output path is empty. Check your config file.");
        paths_valid = false;
    } else {
        ROS_INFO("Output path: %s", params.output_path.c_str());
    }

    if (!paths_valid) {
        ROS_ERROR("Required paths are missing. Please check your configuration file: %s", config_file.c_str());
        return -1;
    }

    // Determine operating mode from parameters
    bool run_preprocessing = false;
    nh.param<bool>("preprocessing", run_preprocessing, false);

    // Check for visualization tools
    std::string viewer_command = "pcl_viewer";
    nh.param<std::string>("visualization_tool", viewer_command, "pcl_viewer");

    if (!viewer_command.empty()) {
        g_calib_processor->setVisualizationTool(viewer_command);
    }

    // Run the appropriate function based mode
    bool success = false;
    try {
        if (run_preprocessing) {
            ROS_INFO("Running preprocessing mode...");
            success = g_calib_processor->runPreprocessing();
        } else {
            ROS_INFO("Running main processing mode...");
            success = g_calib_processor->run();
        }
    } catch (const std::exception &e) {
        ROS_ERROR("Exception during processing: %s", e.what());
        success = false;
    }

    // 清理资源，顺序很重要
    try {
        ROS_INFO("Cleaning up resources...");

        // 先清理自定义资源
        if (g_calib_processor) {
            g_calib_processor.reset();
        }

        // 给ROS一点时间来清理
        ros::Duration(0.1).sleep();

        // 最后关闭ROS
        if (ros::ok()) {
            ros::shutdown();
        }

    } catch (const std::exception &e) {
        ROS_WARN("Exception during cleanup: %s", e.what());
    }

    ROS_INFO("Node shutdown complete");
    return success ? 0 : -1;
}
