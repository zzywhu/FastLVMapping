#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <memory>
#include "signal_handler.h"

#include "config.h"
// Add ROS headers
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <tf2_ros/transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <opencv2/ccalib/omnidir.hpp>

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
     * @brief Constructor with ROS node handle
     * @param nh ROS node handle for publishing data
     */
    CalibProcessor(ros::NodeHandle *nh);

    /**
     * @brief Destructor
     */
    ~CalibProcessor();

    /**
     * @brief Initialize calibration with a config file
     * @param config_file Path to config file
     * @return True if successful
     */
    bool initialize(const std::string &config_file);

    /**
     * @brief Set visualization tool path
     * @param tool_path Path to visualization tool
     */
    void setVisualizationTool(const std::string &tool_path);

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
     * @param output_cloud_ptr Pointer to store colored cloud for emergency save
     * @param output_color_count_ptr Pointer to store color count for emergency save
     * @return true if processing was successful, false otherwise
     */
    bool processImagesAndPointCloud(const std::string &image_folder,
                                    const std::string &trajectory_file,
                                    const std::string &pcd_file,
                                    const std::string &output_folder,
                                    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> *output_cloud_ptr = nullptr,
                                    std::vector<int> **output_color_count_ptr = nullptr);

    // Keep the original method for backward compatibility
    bool processImagesAndPointCloud(const std::string &image_folder,
                                    const std::string &trajectory_file,
                                    const std::string &pcd_file,
                                    const std::string &output_folder) {
        return processImagesAndPointCloud(image_folder, trajectory_file, pcd_file, output_folder, nullptr, nullptr);
    }

    /**
     * @brief Preprocess images and point cloud data for further processing
     * 
     * @param image_folder Folder containing camera images
     * @param trajectory_file File containing LiDAR trajectory
     * @param pcd_file Point cloud file to process
     * @param output_folder Folder for output results
     * @return true if preprocessing was successful, false otherwise
     */
    bool preprocess(const std::string &image_folder,
                    const std::string &trajectory_file,
                    const std::string &pcd_file,
                    const std::string &output_folder);

private:
    // Processing steps
    bool loadCalibrationParameters(const std::string &config_file);
    bool saveCalibrationParameters(const std::string &config_file);
    bool loadDatasets();
    bool preprocess();                 // Keep the no-parameter version for internal use
    bool processImagesAndPointCloud(); // Keep the no-parameter version for internal use
    bool colorizeWithFisheyeImages();

    // Helper methods
    bool interpolatePose(double timestamp,
                         const std::vector<std::pair<double, Eigen::Matrix4d>> &trajectory,
                         Eigen::Matrix4d &lidar_pose);

    bool updateExtrinsics(const std::string &match_file,
                          const std::string &index_file,
                          const Eigen::Matrix4d &lidar_pose,
                          const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud);

    bool colorizePointCloud(const cv::Mat &undistorted_img,
                            const Eigen::Matrix4d &camera_pose,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored_cloud,
                            std::vector<int> &point_color_count,
                            const std::string &output_folder,
                            double timestamp);

    // Add overloaded version with colored_in_this_frame parameter
    bool colorizePointCloud(const cv::Mat &undistorted_img,
                            const Eigen::Matrix4d &camera_pose,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored_cloud,
                            std::vector<int> &point_color_count,
                            const std::string &output_folder,
                            double timestamp,
                            std::vector<bool> &colored_in_this_frame);

    bool processImageFrame(double timestamp,
                           const std::string &img_path,
                           const Eigen::Matrix4d &lidar_pose,
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored_cloud,
                           std::vector<int> &point_color_count,
                           const std::string &output_folder);

    // Add overloaded version with colored_in_this_frame parameter
    bool processImageFrame(double timestamp,
                           const std::string &img_path,
                           const Eigen::Matrix4d &lidar_pose,
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored_cloud,
                           std::vector<int> &point_color_count,
                           const std::string &output_folder,
                           std::vector<bool> &colored_in_this_frame);

    bool saveVisualization(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored_cloud,
                           const std::vector<int> &point_color_count,
                           const std::string &output_folder,
                           const std::string &progress_info);

    // New methods for generating additional outputs
    bool saveOptimizedCameraPoses(const std::vector<std::pair<double, Eigen::Matrix4d>> &trajectory,
                                  const std::string &output_folder);

    bool saveOptimizedExtrinsics(const std::vector<std::pair<double, Eigen::Matrix4d>> &trajectory,
                                 const std::string &output_folder);

    bool createCameraTrajectoryVisualization(const std::vector<std::pair<double, Eigen::Matrix4d>> &trajectory,
                                             const std::string &output_folder);

    // New method for incremental extrinsics saving
    bool saveExtrinsicsForTimestamp(double timestamp, const std::string &output_folder);

    // New ROS-related methods
    void initializeROS();
    void publishColoredPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, double timestamp);
    void publishCameraPose(const Eigen::Matrix4d &camera_pose, double timestamp);
    void publishMatchImage(const std::string &match_image_path, double timestamp);

    // Data members
    cv::Mat camera_matrix_;
    cv::Mat newcamera_matrix_;
    cv::Mat resizecamera_matrix_;
    cv::Mat dist_coeffs_;
    std::string camera_model_; // Added camera model string

    Eigen::Matrix4d T_lidar_camera_;        // Initial LiDAR to camera transformation
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

    // New member variables to track current timestamp and output folder
    double current_timestamp_;

    // New ROS members
    ros::NodeHandle *nh_;
    ros::Publisher pointcloud_pub_;
    ros::Publisher camera_pose_pub_;
    ros::Publisher camera_trajectory_pub_;
    image_transport::Publisher match_image_pub_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    std::string frame_id_;
    bool ros_initialized_;

    // Camera trajectory storage
    std::vector<geometry_msgs::PoseStamped> camera_trajectory_;

    // Variables for processing and tracking state
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame_colored_cloud; // Add this line
};

} // namespace lvmapping
