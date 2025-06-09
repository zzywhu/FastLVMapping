#ifndef CALIB_H
#define CALIB_H

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "ImageProcess/imageprocess.h"

class CalibProcessor {
public:
    CalibProcessor();
    ~CalibProcessor();

    /**
     * Process fisheye images, undistort them, interpolate camera poses,
     * and project point cloud onto each image
     * 
     * @param image_folder Path to directory containing fisheye images
     * @param trajectory_file Path to trajectory file (time, x, y, z, qx, qy, qz, qw)
     * @param pcd_file Path to point cloud file
     * @param output_folder Path to output directory for projected images
     * @return Success flag
     */
    bool processImagesAndPointCloud(const std::string& image_folder, 
                                   const std::string& trajectory_file,
                                   const std::string& pcd_file,
                                   const std::string& output_folder);

    // Add new method to set visualization tool
    void setVisualizationTool(const std::string& tool) { visualization_tool_ = tool; }

private:
    // Helper function to read calibration parameters from config
    bool loadCalibrationParameters(const std::string& config_file);

    // Camera intrinsic parameters
    cv::Mat camera_matrix_;
    cv::Mat newcamera_matrix_;
    cv::Mat dist_coeffs_;
    
    // Extrinsics: transform from LiDAR to camera
    Eigen::Matrix4d T_lidar_camera_;
    
    // Config file path
    std::string config_path_;

    std::string visualization_tool_ = "pcl_viewer";  // Default visualization tool
};

#endif // CALIB_H



