#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace lvmapping {

/**
 * @class Config
 * @brief Centralized parameter management system for the entire project
 */
class Config {
public:
    // Singleton pattern
    static Config &getInstance();

    // Configuration handling
    bool loadFromYAML(const std::string &config_file);
    void saveToYAML(const std::string &config_file) const;

    // Camera parameters
    struct CameraParams {
        cv::Mat camera_matrix;        // Original camera intrinsic matrix
        cv::Mat new_camera_matrix;    // New camera matrix after undistortion
        cv::Mat resize_camera_matrix; // Camera matrix for resized images
        cv::Mat distortion_coeffs;    // Distortion coefficients
        std::string camera_model;     // Camera model: "fisheye" or "pinhole"
    };

    // Calibration parameters
    struct CalibParams {
        Eigen::Matrix4d T_lidar_camera; // Transformation from LiDAR to camera
    };

    // Point cloud processing parameters
    struct PointCloudParams {
        float min_depth;                     // Minimum valid depth (meters)
        float max_depth;                     // Maximum valid depth (meters)
        int neighborhood_size;               // Size of pixel neighborhood for depth checking
        float sky_invalid_ratio_threshold;   // Threshold for invalid depth ratio in sky detection
        float depth_discontinuity_threshold; // Threshold for depth discontinuity
        int gradient_kernel_size;            // Kernel size for gradient calculation
        int sky_detection_kernel_size;       // Kernel size for sky detection
    };

    // Projection parameters
    struct ProjectionParams {
        int image_width;         // Width of undistorted image
        int image_height;        // Height of undistorted image
        int focal_length;        // Focal length for pinhole projection
        int image_center_x;      // Image center X for pinhole projection
        int image_center_y;      // Image center Y for pinhole projection
        int valid_image_start_x; // Start X of valid region in the image
        int valid_image_start_y; // Start Y of valid region in the image
        int valid_image_end_x;   // End X of valid region in the image
        int valid_image_end_y;   // End Y of valid region in the image
    };

    // PnP parameters
    struct PnPParams {
        double max_rotation_diff;    // Maximum allowed rotation difference
        double max_translation_diff; // Maximum allowed translation difference
        double reprojection_error;   // Reprojection error threshold for RANSAC
        int min_inlier_count;        // Minimum number of inliers required
        int ransac_iterations;       // Number of RANSAC iterations
        double confidence;           // RANSAC confidence level
    };

    // Output parameters
    struct OutputParams {
        std::string visualization_tool; // Path to visualization tool executable
        int viz_frequency;              // How often to update visualization
    };

    // Processing parameters
    struct ProcessingParams {
        std::string img_path;
        std::string traj_path;
        std::string pcd_path;
        std::string output_path;
        int img_sampling_step;
        double image_downscale_factor; // Add this field to fix compilation error
        bool use_extrinsic_optimization;
    };

    // Getters for parameter groups
    const CameraParams &cameraParams() const {
        return camera_params_;
    }
    const CalibParams &calibParams() const {
        return calib_params_;
    }
    const PointCloudParams &pointCloudParams() const {
        return pc_params_;
    }
    const ProjectionParams &projectionParams() const {
        return proj_params_;
    }
    const PnPParams &pnpParams() const {
        return pnp_params_;
    }
    const OutputParams &outputParams() const {
        return output_params_;
    }
    const ProcessingParams &processingParams() const {
        return processing_params_;
    }

    // Mutable getters for parameter groups
    CameraParams &mutableCameraParams() {
        return camera_params_;
    }
    CalibParams &mutableCalibParams() {
        return calib_params_;
    }
    PointCloudParams &mutablePointCloudParams() {
        return pc_params_;
    }
    ProjectionParams &mutableProjectionParams() {
        return proj_params_;
    }
    PnPParams &mutablePnpParams() {
        return pnp_params_;
    }
    OutputParams &mutableOutputParams() {
        return output_params_;
    }
    ProcessingParams &mutableProcessingParams() {
        return processing_params_;
    }

private:
    Config(); // Private constructor for singleton pattern

    // Parameter storage
    CameraParams camera_params_;
    CalibParams calib_params_;
    PointCloudParams pc_params_;
    ProjectionParams proj_params_;
    PnPParams pnp_params_;
    OutputParams output_params_;
    ProcessingParams processing_params_;
    void setDefaultValues();
};

} // namespace lvmapping
