#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>  // Added specific point cloud header
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <memory>

namespace lvmapping {

/**
 * @class CalibProcessor
 * @brief Main processor for LiDAR-Camera calibration and point cloud colorization
 */
class CalibProcessor {
public:
    /**
     * @brief Default constructor
     */
    CalibProcessor();
    
    /**
     * @brief Destructor
     */
    ~CalibProcessor();
    
    /**
     * @brief Initialize calibration with a config file
     * @param config_file Path to config file
     * @return True if successful
     */
    bool initialize(const std::string& config_file);
    
    /**
     * @brief Set visualization tool path
     * @param tool_path Path to visualization tool
     */
    void setVisualizationTool(const std::string& tool_path);
    
    /**
     * @brief Main processing function - run all calibration and colorization steps
     * @return True if successful
     */
    bool run();
    
    /**
     * @brief Run preprocessing only (generate index maps and projections)
     * @return True if successful
     */
    bool runPreprocessing();
    
    /**
     * @brief Process images and point cloud to update extrinsic calibration and colorize point cloud
     * @param image_folder Folder containing camera images
     * @param trajectory_file File containing LiDAR trajectory
     * @param pcd_file Point cloud file to colorize
     * @param output_folder Folder for output results
     * @return true if processing was successful, false otherwise
     */
    bool processImagesAndPointCloud(const std::string& image_folder,
                                   const std::string& trajectory_file,
                                   const std::string& pcd_file,
                                   const std::string& output_folder);

    /**
     * @brief Preprocess images and point cloud data for further processing
     * 
     * @param image_folder Folder containing camera images
     * @param trajectory_file File containing LiDAR trajectory
     * @param pcd_file Point cloud file to process
     * @param output_folder Folder for output results
     * @return true if preprocessing was successful, false otherwise
     */
    bool preprocess(const std::string& image_folder,
                   const std::string& trajectory_file,
                   const std::string& pcd_file,
                   const std::string& output_folder);

private:
    // Processing steps
    bool loadCalibrationParameters(const std::string& config_file);
    bool saveCalibrationParameters(const std::string& config_file);
    bool loadDatasets();
    bool preprocess(); // Keep the no-parameter version for internal use
    bool processImagesAndPointCloud(); // Keep the no-parameter version for internal use
    bool colorizeWithFisheyeImages();

    // Helper methods  
    bool interpolatePose(double timestamp, 
                        const std::vector<std::pair<double, Eigen::Matrix4d>>& trajectory,
                        Eigen::Matrix4d& lidar_pose);
    
    bool updateExtrinsics(const std::string& match_file,
                         const std::string& index_file, 
                         const Eigen::Matrix4d& lidar_pose,
                         const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
                         
    bool colorizePointCloud(const cv::Mat& undistorted_img,
                           const Eigen::Matrix4d& camera_pose,
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colored_cloud,
                           std::vector<int>& point_color_count,
                           const std::string& output_folder,
                           double timestamp);
    
    bool processImageFrame(double timestamp, 
                          const std::string& img_path, 
                          const Eigen::Matrix4d& lidar_pose,
                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colored_cloud,
                          std::vector<int>& point_color_count,
                          const std::string& output_folder);
                          
    bool saveVisualization(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colored_cloud,
                          const std::vector<int>& point_color_count,
                          const std::string& output_folder,
                          const std::string& progress_info);
    
    // Data members
    cv::Mat camera_matrix_;
    cv::Mat newcamera_matrix_;
    cv::Mat resizecamera_matrix_;
    cv::Mat dist_coeffs_;
    
    Eigen::Matrix4d T_lidar_camera_;      // Initial LiDAR to camera transformation
    Eigen::Matrix4d T_lidar_camera_update_; // Updated LiDAR to camera transformation
    
    std::string visualization_tool_;
    
    // Dataset holders
    std::vector<std::pair<double, Eigen::Matrix4d>> trajectory_;
    std::vector<std::pair<double, std::string>> image_files_;
    std::vector<std::pair<double, std::string>> selected_images_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_;
    
    // Parameters
    std::string config_file_;
    std::string image_folder_;
    std::string trajectory_file_;
    std::string pcd_file_;
    std::string output_folder_;
};

} // namespace lvmapping



