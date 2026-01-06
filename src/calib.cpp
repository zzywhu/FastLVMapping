#include "calib.h"
#include "ImageProcess/imageprocess.h"
#include "signal_handler.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nav_msgs/Path.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h> // Add this for toRosMsg
#include <std_msgs/ColorRGBA.h>
#include <thread>

namespace fs = std::filesystem;
namespace lvmapping {

bool compareTimestamps(const std::pair<double, Eigen::Matrix4d> &a,
                       const std::pair<double, Eigen::Matrix4d> &b) {
    return a.first < b.first;
}

CalibProcessor::CalibProcessor() {
    // Default initialization using configuration
    Config &config = Config::getInstance();

    camera_matrix_ = config.cameraParams().camera_matrix.clone();
    newcamera_matrix_ = config.cameraParams().new_camera_matrix.clone();
    resizecamera_matrix_ = config.cameraParams().resize_camera_matrix.clone();
    dist_coeffs_ = config.cameraParams().distortion_coeffs.clone();
    T_lidar_camera_ = config.calibParams().T_lidar_camera;
    T_lidar_camera_update_ = T_lidar_camera_; // Initialize with the same value

    // Initialize the point cloud for frame-specific colored points
    frame_colored_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
}

// Add ROS constructor
CalibProcessor::CalibProcessor(ros::NodeHandle *nh) :
    nh_(nh), ros_initialized_(false) {
    // Default initialization using configuration
    Config &config = Config::getInstance();

    camera_matrix_ = config.cameraParams().camera_matrix.clone();
    newcamera_matrix_ = config.cameraParams().new_camera_matrix.clone();
    resizecamera_matrix_ = config.cameraParams().resize_camera_matrix.clone();
    dist_coeffs_ = config.cameraParams().distortion_coeffs.clone();
    T_lidar_camera_ = config.calibParams().T_lidar_camera;
    T_lidar_camera_update_ = T_lidar_camera_; // Initialize with the same value

    // Initialize the point cloud for frame-specific colored points
    frame_colored_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

    // Initialize ROS publishers if we have a valid node handle
    if (nh_) {
        initializeROS();
    }
}

CalibProcessor::~CalibProcessor() {
    // Nothing specific to do in destructor
}

void CalibProcessor::initializeROS() {
    if (!nh_ || ros_initialized_) {
        return;
    }

    // Set up ROS publishers
    pointcloud_pub_ =
        nh_->advertise<sensor_msgs::PointCloud2>("colored_pointcloud", 1);
    camera_pose_pub_ =
        nh_->advertise<visualization_msgs::MarkerArray>("camera_pose", 1);
    camera_trajectory_pub_ =
        nh_->advertise<nav_msgs::Path>("camera_trajectory", 1);

    // Set up image transport publisher
    image_transport::ImageTransport it(*nh_);
    match_image_pub_ = it.advertise("match_visualization", 1);

    depth_image_pub_ = it.advertise("depth_visualization", 1);

    image_pub_ = it.advertise("input_image", 1);

    // Set default frame ID - Make sure this matches what's in your RViz config
    frame_id_ = "map"; // This should match the Fixed Frame in RViz

    ros_initialized_ = true;

    // Initialize empty trajectory
    camera_trajectory_.clear();

    ROS_INFO("ROS publishers initialized for visualization");
}

bool CalibProcessor::loadCalibrationParameters(const std::string &config_file) {
    Config &config = Config::getInstance();
    if (!config.loadFromYAML(config_file)) {
        std::cerr << "Failed to load configuration from " << config_file
                  << std::endl;
        return false;
    }

    // Update member variables from loaded config
    camera_matrix_ = config.cameraParams().camera_matrix.clone();
    newcamera_matrix_ = config.cameraParams().new_camera_matrix.clone();
    resizecamera_matrix_ = config.cameraParams().resize_camera_matrix.clone();
    dist_coeffs_ = config.cameraParams().distortion_coeffs.clone();
    camera_model_ = config.cameraParams().camera_model;

    T_lidar_camera_ = config.calibParams().T_lidar_camera;
    T_lidar_camera_update_ = T_lidar_camera_;

    visualization_tool_ = config.outputParams().visualization_tool;

    std::cout << "Loaded camera model: " << camera_model_ << std::endl;

    return true;
}

bool CalibProcessor::saveCalibrationParameters(const std::string &config_file) {
    Config &config = Config::getInstance();

    // Update config with current values
    config.mutableCameraParams().camera_matrix = camera_matrix_.clone();
    config.mutableCameraParams().new_camera_matrix = newcamera_matrix_.clone();
    config.mutableCameraParams().resize_camera_matrix =
        resizecamera_matrix_.clone();
    config.mutableCameraParams().distortion_coeffs = dist_coeffs_.clone();

    config.mutableCalibParams().T_lidar_camera = T_lidar_camera_update_;

    config.saveToYAML(config_file);
    return true;
}

void CalibProcessor::setVisualizationTool(const std::string &tool_path) {
    visualization_tool_ = tool_path;
    Config::getInstance().mutableOutputParams().visualization_tool = tool_path;
}

bool CalibProcessor::interpolatePose(
    double timestamp,
    const std::vector<std::pair<double, Eigen::Matrix4d>> &trajectory,
    Eigen::Matrix4d &lidar_pose) {
    // Initialize to identity matrix
    lidar_pose = Eigen::Matrix4d::Identity();

    // Check if timestamp is within trajectory range
    if (trajectory.empty()) {
        std::cerr << "Trajectory is empty" << std::endl;
        return false;
    }

    if (timestamp <= trajectory.front().first || timestamp >= trajectory.back().first) {
        return false;
    }

    // Find the trajectory points that surround the image timestamp
    size_t idx = 0;
    while (idx < trajectory.size() - 1 && trajectory[idx + 1].first < timestamp) {
        idx++;
    }

    // Interpolate the pose
    double t1 = trajectory[idx].first;
    double t2 = trajectory[idx + 1].first;
    Eigen::Matrix4d pose1 = trajectory[idx].second;
    Eigen::Matrix4d pose2 = trajectory[idx + 1].second;

    // Linear interpolation factor
    double alpha = (timestamp - t1) / (t2 - t1);

    // Interpolate position
    Eigen::Vector3d pos1 = pose1.block<3, 1>(0, 3);
    Eigen::Vector3d pos2 = pose2.block<3, 1>(0, 3);
    Eigen::Vector3d pos = pos1 + alpha * (pos2 - pos1);

    // Interpolate rotation using quaternions
    Eigen::Quaterniond q1(pose1.block<3, 3>(0, 0));
    Eigen::Quaterniond q2(pose2.block<3, 3>(0, 0));
    Eigen::Quaterniond q = q1.slerp(alpha, q2);

    lidar_pose.block<3, 3>(0, 0) = q.toRotationMatrix();
    lidar_pose.block<3, 1>(0, 3) = pos;

    return true;
}

// Implementation of updateExtrinsics
bool CalibProcessor::updateExtrinsics(
    const std::string &match_file, const std::string &index_file,
    const Eigen::Matrix4d &lidar_pose,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) {
    std::cout << "Using match file: " << match_file
              << " and index file: " << index_file << " to update extrinsics"
              << std::endl;

    // Get configuration parameters
    const Config &config = Config::getInstance();
    const auto &pnp_params = config.pnpParams();

    // Read match file
    std::vector<std::pair<cv::Point2f, cv::Point2f>> matches;
    std::ifstream match_in(match_file);
    if (!match_in.is_open()) {
        std::cerr << "Failed to open match file: " << match_file << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(match_in, line)) {
        std::replace(line.begin(), line.end(), ',',
                     ' '); // Replace commas with spaces
        std::istringstream iss(line);
        float x1, y1, x2, y2;
        if (!(iss >> x1 >> y1 >> x2 >> y2)) {
            continue;
        }
        matches.push_back({cv::Point2f(x1, y1), cv::Point2f(x2, y2)});
    }
    match_in.close();

    // Read index file
    cv::Mat index_map;
    std::ifstream index_in(index_file, std::ios::binary);
    if (!index_in.is_open()) {
        std::cerr << "Failed to open index file: " << index_file << std::endl;
        return false;
    }

    int rows, cols;
    index_in.read(reinterpret_cast<char *>(&rows), sizeof(int));
    index_in.read(reinterpret_cast<char *>(&cols), sizeof(int));

    index_map = cv::Mat(rows, cols, CV_32S);
    index_in.read(reinterpret_cast<char *>(index_map.data),
                  rows * cols * sizeof(int));
    index_in.close();

    // Calculate camera pose from LiDAR pose and extrinsics
    Eigen::Matrix4d camera_pose = lidar_pose * T_lidar_camera_.inverse();

    // Transform point cloud to camera coordinate system
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        const auto &point = cloud->points[i];
        Eigen::Vector4d pt_world(point.x, point.y, point.z, 1.0);
        Eigen::Vector4d pt_camera = camera_pose.inverse() * pt_world;

        pcl::PointXYZI transformed_point;
        transformed_point.x = pt_camera[0];
        transformed_point.y = pt_camera[1];
        transformed_point.z = pt_camera[2];

        transformed_cloud->push_back(transformed_point);
    }

    // Collect 3D-2D correspondences
    std::vector<cv::Point3f> obj_points;
    std::vector<cv::Point2f> img_points;

    for (const auto &match : matches) {
        int x1 = static_cast<int>(match.first.x);
        int y1 = static_cast<int>(match.first.y);

        if (x1 >= 0 && x1 < index_map.cols && y1 >= 0 && y1 < index_map.rows) {
            int point_idx = index_map.at<int>(y1, x1);

            if (point_idx >= 0 && point_idx < transformed_cloud->points.size()) {
                const auto &point = transformed_cloud->points[point_idx];
                obj_points.push_back(cv::Point3f(point.x, point.y, point.z));
                img_points.push_back(match.second);
            }
        }
    }

    if (obj_points.size() < pnp_params.min_inlier_count) {
        std::cout << "Not enough valid correspondences for PnP: "
                  << obj_points.size() << std::endl;
        return false;
    }

    std::cout << "Found " << obj_points.size()
              << " valid 3D-2D correspondences for PnP" << std::endl;

    // Solve PnP
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat inliers;

    bool pnp_success = cv::solvePnPRansac(
        obj_points, img_points, resizecamera_matrix_,
        cv::Mat::zeros(1, 5, CV_64F), rvec, tvec, true,
        pnp_params.ransac_iterations, pnp_params.reprojection_error,
        pnp_params.confidence, inliers, cv::SOLVEPNP_ITERATIVE);
    if (pnp_success && inliers.rows > pnp_params.min_inlier_count) {
        std::cout << "PnP successful with " << inliers.rows << " inliers out of "
                  << obj_points.size() << " points" << std::endl;

        // Convert to rotation matrix
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        // Build the transformation matrix
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;

        for (int i = 0; i < 3; i++) {
            translation(i) = tvec.at<double>(i);
            for (int j = 0; j < 3; j++) {
                rotation(i, j) = R.at<double>(i, j);
            }
        }

        // Build camera to LiDAR transformation matrix
        Eigen::Matrix4d T_delta = Eigen::Matrix4d::Identity();
        T_delta.block<3, 3>(0, 0) = rotation;
        T_delta.block<3, 1>(0, 3) = translation;

        // Check if transformation change is reasonable
        double rotation_diff = (rotation - Eigen::Matrix3d::Identity()).norm();
        double translation_diff = translation.norm();

        std::cout << "Transformation change - Rotation: " << rotation_diff
                  << ", Translation: " << translation_diff << std::endl;

        if (rotation_diff > pnp_params.max_rotation_diff || translation_diff > pnp_params.max_translation_diff) {
            std::cout
                << "Warning: Transformation change too large, skipping this update!"
                << std::endl;
            return false;
        }

        // Update LiDAR to camera transformation matrix
        T_lidar_camera_update_ = T_delta * T_lidar_camera_;

        std::cout << "Updated extrinsics from PnP:" << std::endl;
        std::cout << T_lidar_camera_update_ << std::endl;

        // Save the updated extrinsics immediately for this timestamp
        saveExtrinsicsForTimestamp(current_timestamp_, output_folder_);

        // Check if there's a corresponding match visualization image and publish it
        std::string match_viz_dir = output_folder_ + "/match/visualizations";
        std::string match_viz_path = match_viz_dir + "/" + std::to_string(current_timestamp_) + "_matches.jpg";
        if (ros_initialized_ && std::filesystem::exists(match_viz_path)) {
            publishMatchImage(match_viz_path, current_timestamp_);
        }

        return true;
    } else {
        std::cerr << "PnP failed" << std::endl;
        return false;
    }
}

// Add a new function to save extrinsics for a specific timestamp
bool CalibProcessor::saveExtrinsicsForTimestamp(
    double timestamp, const std::string &output_folder) {
    // Create directory for calibration if it doesn't exist
    std::string calib_dir = output_folder + "/calibration";
    if (!fs::exists(calib_dir)) {
        fs::create_directories(calib_dir);
    }

    // Path to the extrinsics file
    std::string extrinsics_file_path = calib_dir + "/optimized_extrinsics.txt";

    // Check if file exists to determine whether to write header
    bool write_header = !fs::exists(extrinsics_file_path);

    // Open file in append mode if it exists, or create new file
    std::ofstream extr_file;
    if (write_header) {
        extr_file.open(extrinsics_file_path);
        // Write a header line
        extr_file << "# timestamp tx ty tz qx qy qz qw" << std::endl;
    } else {
        extr_file.open(extrinsics_file_path, std::ios::app);
    }

    if (!extr_file.is_open()) {
        std::cerr << "Failed to open optimized extrinsics file: "
                  << extrinsics_file_path << std::endl;
        return false;
    }

    // Extract translation from extrinsics
    Eigen::Vector3d translation = T_lidar_camera_update_.block<3, 1>(0, 3);

    // Extract rotation as quaternion
    Eigen::Matrix3d rot = T_lidar_camera_update_.block<3, 3>(0, 0);
    Eigen::Quaterniond q(rot);
    q.normalize();

    // Write timestamp, position, and quaternion to file
    extr_file << std::fixed << std::setprecision(9) << timestamp << " "
              << std::setprecision(6) << translation.x() << " " << translation.y()
              << " " << translation.z() << " " << q.x() << " " << q.y() << " "
              << q.z() << " " << q.w() << std::endl;

    extr_file.close();
    std::cout << "Saved optimized extrinsics for timestamp " << timestamp
              << " to: " << extrinsics_file_path << std::endl;

    return true;
}
bool CalibProcessor::colorizePointCloud(
    const cv::Mat &undistorted_img, const Eigen::Matrix4d &camera_pose,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored_cloud,
    std::vector<int> &point_color_count, const std::string &output_folder,
    double timestamp, std::vector<bool> &colored_in_this_frame) {
    // Get configuration parameters
    const Config &config = Config::getInstance();
    const auto &pc_params = config.pointCloudParams();
    const auto &proj_params = config.projectionParams();

    // Create depth visualization image
    cv::Mat depth_viz_img = undistorted_img.clone();

    // 创建视锥体过滤
    Eigen::Matrix4d world_to_camera = camera_pose.inverse();

    // 创建深度缓冲区和点索引缓冲区
    cv::Mat depth_buffer(depth_viz_img.size(), CV_32F,
                         cv::Scalar(std::numeric_limits<float>::max()));
    cv::Mat point_index_buffer(depth_viz_img.size(), CV_32S, cv::Scalar(-1));

    // 新增：创建深度梯度检测缓冲区
    cv::Mat depth_gradient_buffer(depth_viz_img.size(), CV_32F, cv::Scalar(0.0f));

    // 统计跳过的已着色点数量
    std::atomic<int> skipped_colored_points(0);
    std::atomic<int> filtered_valid_points(0);

    // 优化步骤1：预分配空间以减少锁竞争
    std::vector<pcl::PointXYZI> filtered_points;
    std::vector<int> filtered_indices;
    filtered_points.reserve(cloud->size() / 4);
    filtered_indices.reserve(cloud->size() / 4);

    // 优化步骤2：使用更大的块大小，让每个线程处理更多连续的点
    const size_t block_size = 4096;

// 优化步骤3：先过滤所有点，分成两阶段来减少线程同步和临界区（跳过已着色点）
#pragma omp parallel if (OPENMP_FOUND)
    {
        // 每个线程局部存储过滤后的点，避免频繁的锁竞争
        std::vector<pcl::PointXYZI> local_filtered_points;
        std::vector<int> local_filtered_indices;
        local_filtered_points.reserve(block_size);
        local_filtered_indices.reserve(block_size);

        int local_skipped_count = 0;
        int local_valid_count = 0;

// 第一阶段：视锥体过滤（跳过已着色点）
#pragma omp for schedule(dynamic, block_size)
        for (size_t i = 0; i < cloud->points.size(); i++) {
            // **检查点是否已经被着色，如果是则跳过**
            // if (point_color_count[i] > 0) {
            //     local_skipped_count++;
            //     continue;
            // }

            Eigen::Vector4d pt_world(cloud->points[i].x, cloud->points[i].y,
                                     cloud->points[i].z, 1.0);
            Eigen::Vector4d pt_camera = world_to_camera * pt_world;

            // 基本剔除: 点在相机后面或太远
            if (pt_camera[2] <= 0 || pt_camera[2] >= pc_params.max_depth)
                continue;

            // 投影到图像平面
            int px = static_cast<int>(proj_params.focal_length * pt_camera[0] / pt_camera[2] + proj_params.image_center_x);
            int py = static_cast<int>(proj_params.focal_length * pt_camera[1] / pt_camera[2] + proj_params.image_center_y);

            // 检查是否在有效图像边界内
            if (px >= proj_params.valid_image_start_x && px < proj_params.valid_image_end_x && py >= proj_params.valid_image_start_y && py < proj_params.valid_image_end_y) {
                // 添加到本地线程存储
                local_filtered_points.push_back(cloud->points[i]);
                local_filtered_indices.push_back(i);
                local_valid_count++;
            }
        }

        skipped_colored_points += local_skipped_count;
        filtered_valid_points += local_valid_count;

// 合并本地结果到全局容器
#pragma omp critical
        {
            filtered_points.insert(filtered_points.end(),
                                   local_filtered_points.begin(),
                                   local_filtered_points.end());
            filtered_indices.insert(filtered_indices.end(),
                                    local_filtered_indices.begin(),
                                    local_filtered_indices.end());
        }
    }

    std::cout << "Frame " << timestamp << " filtering:" << std::endl;
    std::cout << "  - Skipped already colored points: " << skipped_colored_points.load() << std::endl;
    std::cout << "  - Valid points for processing: " << filtered_valid_points.load() << std::endl;

    // 如果没有有效点需要处理，直接返回
    if (filtered_points.empty()) {
        std::cout << "  - No new points to process for this frame" << std::endl;
        return false;
    }

    // 优化步骤4：批量构建深度缓冲区，减少锁竞争
    struct PointProjInfo {
        int original_idx;
        int px, py;
        float depth;
    };

    std::vector<PointProjInfo> proj_points;
    proj_points.reserve(filtered_points.size());

    // 计算所有过滤后点的投影信息
    for (size_t i = 0; i < filtered_points.size(); i++) {
        const auto &point = filtered_points[i];
        int original_idx = filtered_indices[i];

        Eigen::Vector4d pt_world(point.x, point.y, point.z, 1.0);
        Eigen::Vector4d pt_camera = world_to_camera * pt_world;

        float depth = static_cast<float>(pt_camera[2]);

        // 投影到图像平面
        int px = static_cast<int>(proj_params.focal_length * pt_camera[0] / pt_camera[2] + proj_params.image_center_x);
        int py = static_cast<int>(proj_params.focal_length * pt_camera[1] / pt_camera[2] + proj_params.image_center_y);

        proj_points.push_back({original_idx, px, py, depth});
    }

    // 优化步骤5：使用分块策略更新深度缓冲区
    const int block_width = 64;
    const int block_height = 64;

    int num_blocks_x = (proj_params.valid_image_end_x - proj_params.valid_image_start_x + block_width - 1) / block_width;
    int num_blocks_y = (proj_params.valid_image_end_y - proj_params.valid_image_start_y + block_height - 1) / block_height;
    int total_blocks = num_blocks_x * num_blocks_y;

// 对每个图像块并行处理
#pragma omp parallel for schedule(dynamic) if (OPENMP_FOUND)
    for (int block_idx = 0; block_idx < total_blocks; block_idx++) {
        int block_x = block_idx % num_blocks_x;
        int block_y = block_idx / num_blocks_x;

        int start_x = proj_params.valid_image_start_x + block_x * block_width;
        int start_y = proj_params.valid_image_start_y + block_y * block_height;
        int end_x = std::min(start_x + block_width, proj_params.valid_image_end_x);
        int end_y = std::min(start_y + block_height, proj_params.valid_image_end_y);

        // 处理每个投影点，看它是否影响这个块
        for (const auto &proj_info : proj_points) {
            int px = proj_info.px;
            int py = proj_info.py;

            // 检查点的影响区域是否与当前块重叠
            int min_ny = std::max(start_y, py - pc_params.neighborhood_size);
            int max_ny = std::min(end_y - 1, py + pc_params.neighborhood_size);
            int min_nx = std::max(start_x, px - pc_params.neighborhood_size);
            int max_nx = std::min(end_x - 1, px + pc_params.neighborhood_size);

            if (min_ny > max_ny || min_nx > max_nx)
                continue;

            // 更新重叠区域的深度缓冲区
            for (int ny = min_ny; ny <= max_ny; ny++) {
                for (int nx = min_nx; nx <= max_nx; nx++) {
                    if (proj_info.depth < depth_buffer.at<float>(ny, nx)) {
                        depth_buffer.at<float>(ny, nx) = proj_info.depth;
                        point_index_buffer.at<int>(ny, nx) = proj_info.original_idx;
                    }
                }
            }
        }
    }

    const float depth_discontinuity_threshold = pc_params.depth_discontinuity_threshold; // 深度不连续阈值（米）
    const int gradient_kernel_size = pc_params.gradient_kernel_size;                     // 梯度计算核大小
    const int sky_detection_kernel_size = pc_params.sky_detection_kernel_size;           // 天空检测核大小
    const float sky_invalid_ratio_threshold = pc_params.sky_invalid_ratio_threshold;     // 无效深度比例阈值

#pragma omp parallel for collapse(2) if (OPENMP_FOUND)
    for (int y = std::max(gradient_kernel_size, sky_detection_kernel_size);
         y < depth_buffer.rows - std::max(gradient_kernel_size, sky_detection_kernel_size); y++) {
        for (int x = std::max(gradient_kernel_size, sky_detection_kernel_size);
             x < depth_buffer.cols - std::max(gradient_kernel_size, sky_detection_kernel_size); x++) {
            float center_depth = depth_buffer.at<float>(y, x);

            // 如果中心像素没有有效深度，跳过
            if (center_depth == std::numeric_limits<float>::max()) {
                continue;
            }

            // 1. 计算周围的深度梯度（原有逻辑）
            float max_gradient = 0.0f;

            // 检查3x3邻域内的深度变化
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue; // 跳过中心点

                    int neighbor_x = x + dx;
                    int neighbor_y = y + dy;

                    float neighbor_depth = depth_buffer.at<float>(neighbor_y, neighbor_x);

                    // 如果邻居有有效深度，计算梯度
                    if (neighbor_depth != std::numeric_limits<float>::max()) {
                        float gradient = std::abs(center_depth - neighbor_depth);
                        max_gradient = std::max(max_gradient, gradient);
                    }
                }
            }

            // 2. 新增：检测天空或背景区域（周围大量无效深度）
            int total_neighbors = 0;
            int invalid_neighbors = 0;

            // 在更大的邻域内检查无效深度的比例
            for (int dy = -sky_detection_kernel_size; dy <= sky_detection_kernel_size; dy++) {
                for (int dx = -sky_detection_kernel_size; dx <= sky_detection_kernel_size; dx++) {
                    if (dx == 0 && dy == 0) continue; // 跳过中心点

                    int neighbor_x = x + dx;
                    int neighbor_y = y + dy;

                    total_neighbors++;

                    float neighbor_depth = depth_buffer.at<float>(neighbor_y, neighbor_x);
                    if (neighbor_depth == std::numeric_limits<float>::max()) {
                        invalid_neighbors++;
                    }
                }
            }

            // 计算无效深度比例
            float invalid_ratio = static_cast<float>(invalid_neighbors) / total_neighbors;

            // 3. 新增：检测深度跳跃（前景到背景的突变）
            bool has_depth_jump = false;
            const float depth_jump_threshold = 2.0f; // 深度跳跃阈值（米）

            // 检查是否存在从近距离到远距离的跳跃
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;

                    int neighbor_x = x + dx;
                    int neighbor_y = y + dy;

                    float neighbor_depth = depth_buffer.at<float>(neighbor_y, neighbor_x);

                    if (neighbor_depth != std::numeric_limits<float>::max()) {
                        // 如果邻居深度远大于中心深度，可能是前景到背景的跳跃
                        if (neighbor_depth - center_depth > depth_jump_threshold) {
                            has_depth_jump = true;
                            break;
                        }
                    }
                }
                if (has_depth_jump) break;
            }

            // 综合判断：设置梯度值
            float final_gradient = max_gradient;

            // 如果周围无效深度比例过高，认为是天空
            if (invalid_ratio > sky_invalid_ratio_threshold) {
                final_gradient = std::max(final_gradient, depth_discontinuity_threshold * 1.5f);
            }

            // 如果存在深度跳跃，增加梯度值
            if (has_depth_jump) {
                final_gradient = std::max(final_gradient, depth_discontinuity_threshold * 1.2f);
            }

            depth_gradient_buffer.at<float>(y, x) = final_gradient;
        }
    }

    // 第二阶段：给点云着色并创建可视化（增加深度不连续检测）
    int colored_count = 0;
    std::atomic<int> atomic_colored_count(0);
    std::atomic<int> atomic_skipped_discontinuity(0);

// 优化步骤6：使用图像块并行处理
#pragma omp parallel for collapse(2) schedule(dynamic) if (OPENMP_FOUND)
    for (int by = 0; by < num_blocks_y; by++) {
        for (int bx = 0; bx < num_blocks_x; bx++) {
            int start_x = proj_params.valid_image_start_x + bx * block_width;
            int start_y = proj_params.valid_image_start_y + by * block_height;
            int end_x = std::min(start_x + block_width, proj_params.valid_image_end_x);
            int end_y = std::min(start_y + block_height, proj_params.valid_image_end_y);

            // 块内局部计数器
            int local_colored_count = 0;
            int local_skipped_discontinuity = 0;
            int local_skipped_already_colored = 0;

            // 对块内每个像素处理
            for (int y = start_y; y < end_y; y++) {
                for (int x = start_x; x < end_x; x++) {
                    int point_idx = point_index_buffer.at<int>(y, x);
                    if (point_idx >= 0) {
                        // 获取深度值
                        float depth = depth_buffer.at<float>(y, x);

                        // 新增：检查深度不连续性
                        float depth_gradient = 0.0f;
                        if (y >= gradient_kernel_size && y < depth_gradient_buffer.rows - gradient_kernel_size && x >= gradient_kernel_size && x < depth_gradient_buffer.cols - gradient_kernel_size) {
                            depth_gradient = depth_gradient_buffer.at<float>(y, x);
                        }

                        // 如果深度梯度超过阈值，跳过此点的着色（认为是边缘不连续区域）
                        if (depth_gradient > depth_discontinuity_threshold) {
                            // 在深度可视化中标记为红色（可选，用于调试）
                            // 创建深度可视化
                            float normalized_depth = (depth - pc_params.min_depth) / (pc_params.max_depth - pc_params.min_depth);
                            normalized_depth = std::min(std::max(normalized_depth, 0.0f), 1.0f);

                            int r = static_cast<int>((1.0f - normalized_depth) * 255);
                            int g = static_cast<int>((normalized_depth > 0.5f ? 1.0f - normalized_depth : normalized_depth) * 255);
                            int b = static_cast<int>(normalized_depth * 255);

                            depth_viz_img.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
                            local_skipped_discontinuity++;
                            continue; // 跳过着色
                        }

                        // 创建深度可视化
                        float normalized_depth = (depth - pc_params.min_depth) / (pc_params.max_depth - pc_params.min_depth);
                        normalized_depth = std::min(std::max(normalized_depth, 0.0f), 1.0f);

                        int r = static_cast<int>((1.0f - normalized_depth) * 255);
                        int g = static_cast<int>((normalized_depth > 0.5f ? 1.0f - normalized_depth : normalized_depth) * 255);
                        int b = static_cast<int>(normalized_depth * 255);

                        depth_viz_img.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);

                        // 给点云上色
                        cv::Vec3b color = undistorted_img.at<cv::Vec3b>(y, x);

                        // 更新点云颜色
                        colored_cloud->points[point_idx].b = color[0];
                        colored_cloud->points[point_idx].g = color[1];
                        colored_cloud->points[point_idx].r = color[2];

                        // 标记这个点在此帧中被着色
                        colored_in_this_frame[point_idx] = true;

                        // 递增此点的颜色计数
                        point_color_count[point_idx]++;

                        local_colored_count++;
                    }
                }
            }

            // 更新全局计数（原子操作）
            atomic_colored_count += local_colored_count;
            atomic_skipped_discontinuity += local_skipped_discontinuity;
            //atomic_skipped_already_colored += local_skipped_already_colored;
        }
    }

    // 获取最终着色点数
    colored_count = atomic_colored_count.load();
    int skipped_discontinuity = atomic_skipped_discontinuity.load();

    std::cout << "  - Successfully colored: " << colored_count << " new points" << std::endl;
    std::cout << "  - Skipped (discontinuity): " << skipped_discontinuity << " points" << std::endl;

    // 仅当有点被着色时才保存深度可视化图像
    // ...existing code...

    // 仅当有点被着色时才保存深度可视化图像
    if (colored_count > 0) {
        std::string depth_viz_dir = output_folder + "/depth_viz";
        if (!fs::exists(depth_viz_dir)) {
            fs::create_directories(depth_viz_dir);
        }

        // // 保存伪彩色深度图
        // std::string depth_viz_path = depth_viz_dir + "/" + std::to_string(timestamp) + "_depth.png";
        // publishDepthImage(depth_viz_img, timestamp); // 发布深度图像
        // cv::imwrite(depth_viz_path, depth_viz_img);

        // 保存单通道灰度深度图（越近越亮）
        float min_depth = std::numeric_limits<float>::max();
        float max_depth = std::numeric_limits<float>::lowest();
        for (int y = 0; y < depth_buffer.rows; ++y) {
            for (int x = 0; x < depth_buffer.cols; ++x) {
                float d = depth_buffer.at<float>(y, x);
                if (d != std::numeric_limits<float>::max()) {
                    if (d < min_depth) min_depth = d;
                    if (d > max_depth) max_depth = d;
                }
            }
        }
        // 防止无有效深度
        if (min_depth >= max_depth) {
            min_depth = pc_params.min_depth;
            max_depth = pc_params.max_depth;
        }

        cv::Mat gray_depth(depth_buffer.size(), CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < depth_buffer.rows; ++y) {
            for (int x = 0; x < depth_buffer.cols; ++x) {
                float d = depth_buffer.at<float>(y, x);
                if (d == std::numeric_limits<float>::max()) {
                    gray_depth.at<uchar>(y, x) = 0; // 无效深度为黑色
                } else {
                    // 归一化到[0,1]，越近越亮
                    float norm = 1.0f - (d - min_depth) / (max_depth - min_depth);
                    norm = std::min(std::max(norm, 0.0f), 1.0f);
                    gray_depth.at<uchar>(y, x) = static_cast<uchar>(norm * 255.0f);
                }
            }
        }
        std::string gray_depth_path = depth_viz_dir + "/" + std::to_string(timestamp) + "_depth_gray.png";
        cv::imwrite(gray_depth_path, gray_depth);
        publishDepthImage(gray_depth, timestamp);
        publishImage(undistorted_img, timestamp);

        // ...existing code...
        // **优化的梯度可视化：天空边缘用蓝色，其他边缘用红色**
        cv::Mat simple_gradient_viz = cv::Mat::zeros(depth_viz_img.size(), CV_8UC3);

        int border_size = std::max(gradient_kernel_size, sky_detection_kernel_size);
        const int thick = 3; // 加粗半径，1表示3x3，2表示5x5
        for (int y = border_size; y < depth_buffer.rows - border_size; y++) {
            for (int x = border_size; x < depth_buffer.cols - border_size; x++) {
                float depth_gradient = depth_gradient_buffer.at<float>(y, x);

                if (depth_gradient > depth_discontinuity_threshold) {
                    float center_depth = depth_buffer.at<float>(y, x);
                    cv::Vec3b color(0, 0, 255); // 默认红色
                    if (center_depth != std::numeric_limits<float>::max()) {
                        int total_neighbors = 0;
                        int invalid_neighbors = 0;
                        for (int dy = -sky_detection_kernel_size; dy <= sky_detection_kernel_size; dy++) {
                            for (int dx = -sky_detection_kernel_size; dx <= sky_detection_kernel_size; dx++) {
                                if (dx == 0 && dy == 0) continue;
                                int neighbor_x = x + dx;
                                int neighbor_y = y + dy;
                                if (neighbor_x >= 0 && neighbor_x < depth_buffer.cols && neighbor_y >= 0 && neighbor_y < depth_buffer.rows) {
                                    total_neighbors++;
                                    float neighbor_depth = depth_buffer.at<float>(neighbor_y, neighbor_x);
                                    if (neighbor_depth == std::numeric_limits<float>::max()) {
                                        invalid_neighbors++;
                                    }
                                }
                            }
                        }
                        float invalid_ratio = (total_neighbors > 0) ?
                                                  static_cast<float>(invalid_neighbors) / total_neighbors :
                                                  0.0f;
                        if (invalid_ratio > sky_invalid_ratio_threshold) {
                            color = cv::Vec3b(255, 0, 0); // 蓝色
                        }
                    }
                    // 加粗显示
                    for (int dy = -thick; dy <= thick; dy++) {
                        for (int dx = -thick; dx <= thick; dx++) {
                            int nx = x + dx;
                            int ny = y + dy;
                            if (nx >= 0 && nx < simple_gradient_viz.cols && ny >= 0 && ny < simple_gradient_viz.rows) {
                                simple_gradient_viz.at<cv::Vec3b>(ny, nx) = color;
                            }
                        }
                    }
                }
            }
        }

        std::string simple_gradient_path = depth_viz_dir + "/" + std::to_string(timestamp) + "_gradient_simple.png";
        cv::imwrite(simple_gradient_path, simple_gradient_viz);
        // ...existing code...
    }

    return colored_count > 0;
}
bool CalibProcessor::saveVisualization(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored_cloud,
    const std::vector<int> &point_color_count, const std::string &output_folder,
    const std::string &progress_info) {
    // 创建colormap目录（如果不存在）
    std::string colormap_dir = output_folder + "/colormap";
    if (!fs::exists(colormap_dir)) {
        fs::create_directories(colormap_dir);
    }

    // 显示进度信息
    std::cout << progress_info << std::endl;

    // 统计着色点数
    int colored_points = 0;
#pragma omp parallel for reduction(+ \
                                   : colored_points) if (OPENMP_FOUND)
    for (size_t i = 0; i < point_color_count.size(); ++i) {
        if (point_color_count[i] > 0) {
            colored_points++;
        }
    }

    double percentage = (100.0 * colored_points / colored_cloud->points.size());
    std::cout << "Points colored so far: " << colored_points << "/"
              << colored_cloud->points.size() << " (" << percentage << "%)"
              << std::endl;

    // 如果是第一次保存，提供PCL查看器信息
    static bool viewer_info_shown = false;
    if (!viewer_info_shown) {
        if (!visualization_tool_.empty()) {
            std::cout << "To watch the coloring process in real-time, you can run:"
                      << std::endl;
            std::cout << visualization_tool_ << " " << colormap_dir
                      << "/colored_pointcloud_live.pcd" << std::endl;
        } else {
            std::cout << "No point cloud viewer available. Install pcl-tools, "
                         "CloudCompare, or MeshLab for visualization."
                      << std::endl;
        }
        viewer_info_shown = true;
    }

    // 创建可视化点云 - 只保存当前进度状态，减少保存完整点云的次数
    static int last_save_colored_count = 0;

    // 如果着色点数明显增加(至少增加1%或总数的1000个点)，才保存新的点云
    if (colored_points > last_save_colored_count + std::max(static_cast<int>(colored_cloud->points.size() * 0.01),
                                                            1000)
        || percentage >= 99.9) { // 或者接近完成时保存

        std::cout << "Saving visualization point cloud..." << std::endl;

        // 创建一个可视化点云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr viz_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>);

        // 避免制作完整副本，只添加已着色的点
        for (size_t i = 0; i < colored_cloud->points.size(); ++i) {
            if (point_color_count[i] > 0) {
                pcl::PointXYZRGB colored_point = colored_cloud->points[i];
                viz_cloud->push_back(colored_point);
            }
        }

        // 保存活动可视化文件
        pcl::io::savePCDFileBinary(colormap_dir + "/colored_pointcloud_live.pcd",
                                   *viz_cloud);

        // 更新上次保存的着色点数
        last_save_colored_count = colored_points;

        // 如果完成度超过99.9%，保存最终结果
        if (percentage >= 99.9) {
            std::string final_cloud_path =
                colormap_dir + "/colored_pointcloud_final.pcd";
            pcl::io::savePCDFileBinary(final_cloud_path, *viz_cloud);
            std::cout << "Final colored point cloud saved to: " << final_cloud_path
                      << std::endl;
        }
    }

    return true;
}

bool CalibProcessor::processImageFrame(
    double timestamp, const std::string &img_path,
    const Eigen::Matrix4d &lidar_pose,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored_cloud,
    std::vector<int> &point_color_count, const std::string &output_folder,
    std::vector<bool> &colored_in_this_frame) {
    // 获取图像缩放因子
    const Config &config = Config::getInstance();
    double image_downscale_factor =
        config.processingParams().image_downscale_factor;

    // 加载图像
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to open image: " << img_path << std::endl;
        return false;
    }

    // 获取相机模型
    const std::string &camera_model = config.cameraParams().camera_model;

    // 基于相机模型去除畸变
    cv::Mat undistorted_img;
    if (camera_model == "fisheye") {
        cv::fisheye::undistortImage(img, undistorted_img, camera_matrix_,
                                    dist_coeffs_, newcamera_matrix_);
    } else { // pinhole or default
        cv::undistort(img, undistorted_img, camera_matrix_, dist_coeffs_,
                      newcamera_matrix_);
    }

    // 计算相机位姿
    Eigen::Matrix4d camera_pose = lidar_pose * T_lidar_camera_update_.inverse();

    // 用深度感知方法给点云着色并跟踪哪些点被着色了
    return colorizePointCloud(undistorted_img, camera_pose, cloud, colored_cloud,
                              point_color_count, output_folder, timestamp,
                              colored_in_this_frame);
}

bool CalibProcessor::processImagesAndPointCloud(
    const std::string &image_folder, const std::string &trajectory_file,
    const std::string &pcd_file, const std::string &output_folder,
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> *output_cloud_ptr,
    std::vector<int> **output_color_count_ptr) {
    // 在函数开始时就设置信号处理
    std::atomic<bool> processing_terminated{false};

    // Get configuration parameters
    const Config &config = Config::getInstance();
    const int viz_frequency = config.outputParams().viz_frequency;
    const int img_sampling_step = config.processingParams().img_sampling_step;
    const bool use_optimize = config.processingParams().use_extrinsic_optimization;
    const bool use_time = config.processingParams().use_time;

    std::cout << "Using extrinsic optimization: " << (use_optimize ? "Yes" : "No") << std::endl;

    // Create output directory if it doesn't exist
    if (output_folder.empty()) {
        std::cerr << "Error: Output folder path is empty" << std::endl;
        return false;
    }

    // Create a timer for the entire process
    auto total_start_time = std::chrono::high_resolution_clock::now();

    try {
        if (!fs::exists(output_folder)) {
            if (!fs::create_directories(output_folder)) {
                std::cerr << "Failed to create output directory: " << output_folder << std::endl;
                return false;
            }
        }
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return false;
    }

    // 早期终止检查
    if (SignalHandler::getInstance().shouldTerminate()) {
        std::cout << "Termination requested before processing started." << std::endl;
        return true;
    }

    // Time tracking for trajectory loading
    auto traj_load_start = std::chrono::high_resolution_clock::now();

    // Load trajectory data
    std::vector<std::pair<double, Eigen::Matrix4d>> trajectory;
    std::ifstream traj_file(trajectory_file);
    if (!traj_file.is_open()) {
        std::cerr << "Failed to open trajectory file: " << trajectory_file << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(traj_file, line)) {
        // 在长时间操作中检查终止信号
        if (SignalHandler::getInstance().shouldTerminate()) {
            std::cout << "Termination requested during trajectory loading." << std::endl;
            return true;
        }

        std::istringstream iss(line);
        double time, x, y, z, qx, qy, qz, qw;
        if (!(iss >> time >> x >> y >> z >> qx >> qy >> qz >> qw)) {
            continue;
        }

        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        Eigen::Quaterniond q(qw, qx, qy, qz);
        pose.block<3, 3>(0, 0) = q.toRotationMatrix();
        pose.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);

        trajectory.push_back({time, pose});
    }

    // Sort trajectory by timestamp
    std::sort(trajectory.begin(), trajectory.end(), compareTimestamps);

    auto traj_load_end = std::chrono::high_resolution_clock::now();
    auto traj_load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              traj_load_end - traj_load_start)
                              .count();
    std::cout << "Trajectory loaded with " << trajectory.size() << " poses in "
              << traj_load_time / 1000.0 << " seconds." << std::endl;

    // 检查终止信号
    if (SignalHandler::getInstance().shouldTerminate()) {
        std::cout << "Termination requested after trajectory loading." << std::endl;
        return true;
    }

    // 优化读取点云过程 - 使用带进度显示的点云加载
    std::cout << "Loading point cloud from: " << pcd_file << " (this may take a while)..." << std::endl;
    auto cloud_load_start = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *cloud) == -1) {
        std::cerr << "Failed to load point cloud: " << pcd_file << std::endl;
        return false;
    }

    auto cloud_load_end = std::chrono::high_resolution_clock::now();
    auto cloud_load_time = std::chrono::duration_cast<std::chrono::seconds>(
                               cloud_load_end - cloud_load_start)
                               .count();
    std::cout << "Point cloud loaded with " << cloud->size() << " points in "
              << cloud_load_time << " seconds." << std::endl;

    // 检查终止信号
    if (SignalHandler::getInstance().shouldTerminate()) {
        std::cout << "Termination requested after point cloud loading." << std::endl;
        return true;
    }

    // Time tracking for cloud initialization
    auto cloud_init_start = std::chrono::high_resolution_clock::now();

    // 创建着色点云
    std::cout << "Initializing colored point cloud..." << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored_cloud->points.resize(cloud->points.size());
    colored_cloud->width = cloud->width;
    colored_cloud->height = cloud->height;

// 并行初始化着色点云
#pragma omp parallel for schedule(dynamic, 10000) if (OPENMP_FOUND)
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        colored_cloud->points[i].x = cloud->points[i].x;
        colored_cloud->points[i].y = cloud->points[i].y;
        colored_cloud->points[i].z = cloud->points[i].z;
        colored_cloud->points[i].r = 0;
        colored_cloud->points[i].g = 0;
        colored_cloud->points[i].b = 0;
    }

    auto cloud_init_end = std::chrono::high_resolution_clock::now();
    auto cloud_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                               cloud_init_end - cloud_init_start)
                               .count();
    std::cout << "Colored point cloud initialized in " << cloud_init_time / 1000.0 << " seconds." << std::endl;

    // 为每个点创建计数器
    std::vector<int> point_color_count(cloud->points.size(), 0);

    // Provide access to the colored cloud and count for emergency save
    if (output_cloud_ptr) {
        *output_cloud_ptr = colored_cloud;
    }
    if (output_color_count_ptr) {
        *output_color_count_ptr = &point_color_count;
    }

    // 检查终止信号
    if (SignalHandler::getInstance().shouldTerminate()) {
        std::cout << "Termination requested after initialization." << std::endl;
        return true;
    }

    // Time tracking for image file scanning
    auto img_scan_start = std::chrono::high_resolution_clock::now();

    // Get all image files from directory
    std::vector<std::pair<double, std::string>> image_files;
    for (const auto &entry : fs::directory_iterator(image_folder)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            // Assuming filename is the timestamp
            double timestamp =
                std::stod(filename.substr(0, filename.find_last_of('.')));
            image_files.push_back({timestamp, entry.path().string()});
        }
    }

    // Sort images by timestamp
    std::sort(image_files.begin(), image_files.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    auto img_scan_end = std::chrono::high_resolution_clock::now();
    auto img_scan_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                             img_scan_end - img_scan_start)
                             .count();
    std::cout << "Found and sorted " << image_files.size() << " images in "
              << img_scan_time / 1000.0 << " seconds." << std::endl;

    // Sample images to improve processing speed
    std::vector<std::pair<double, std::string>> selected_images;
    for (size_t i = 0; i < image_files.size();
         i += img_sampling_step) { // Use the already defined img_sampling_step
        selected_images.push_back(image_files[i]);
    }

    std::cout << "Selected " << selected_images.size() << " images out of "
              << image_files.size() << " for processing (sampling every "
              << img_sampling_step << " images)" << std::endl;

    // Time tracking for match/index file scanning
    auto match_scan_start = std::chrono::high_resolution_clock::now();

    // Get match and index file folders
    std::string match_folder = output_folder + "/match/coordinates";
    std::string index_folder = output_folder + "/index_maps";

    if (!fs::exists(match_folder) || !fs::exists(index_folder)) {
        std::cerr
            << "Match or index folders not found. Please run preprocess first."
            << std::endl;
        return false;
    }

    // Collect all match and index files
    std::map<double, std::string> match_files;
    std::map<double, std::string> index_files;

    // Read match files
    for (const auto &entry : fs::directory_iterator(match_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::string filename = entry.path().filename().string();
            double timestamp =
                std::stod(filename.substr(0, filename.find_last_of('.')));
            match_files[timestamp] = entry.path().string();
        }
    }

    // Read index files
    for (const auto &entry : fs::directory_iterator(index_folder)) {
        if (entry.is_regular_file() && entry.path().filename().string().find("_index.bin") != std::string::npos) {
            std::string filename = entry.path().filename().string();
            double timestamp =
                std::stod(filename.substr(0, filename.find_last_of('_')));
            index_files[timestamp] = entry.path().string();
        }
    }

    auto match_scan_end = std::chrono::high_resolution_clock::now();
    auto match_scan_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                               match_scan_end - match_scan_start)
                               .count();
    std::cout << "Found " << match_files.size() << " match files and "
              << index_files.size() << " index files in "
              << match_scan_time / 1000.0 << " seconds." << std::endl;

    // Flag to track if extrinsics were updated
    bool extrinsics_updated = false;

    // Process each selected image
    ros::Rate publish_rate(
        30); // Increase rate to 30Hz for smoother visualization

    // Clear trajectory for new processing run
    if (ros_initialized_) {
        camera_trajectory_.clear();
    }

    // 添加进度显示和预计时间
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Starting image processing..." << std::endl;

    // ===== 新增：渲染与PSNR输出目录与CSV初始化 (增加SSIM) =====
    std::string render_dir = output_folder + "/render";
    if (!fs::exists(render_dir)) {
        fs::create_directories(render_dir);
    }
    std::string psnr_csv_path = render_dir + "/psnr.csv";
    std::ofstream psnr_csv(psnr_csv_path);
    if (psnr_csv.is_open()) {
        psnr_csv << "timestamp,valid_pixels,psnr_db,ssim" << std::endl; // 增加SSIM列
    }

    // Statistics tracking
    double total_colorize_time = 0.0;
    double total_update_extrinsics_time = 0.0;
    double total_ros_pub_time = 0.0;
    double total_interpolate_time = 0.0;
    int colorize_count = 0;
    int extrinsics_update_count = 0;
    int ros_pub_count = 0;
    int img_count = 0;

    // 新增：记录每帧的外参（T_lidar_camera_update_）以便最终完整渲染
    std::map<double, Eigen::Matrix4d> per_frame_extrinsics;

    // 优化的主处理循环
    for (const auto &img_data : selected_images) {
        // 在每次循环开始时立即检查终止信号
        if (SignalHandler::getInstance().shouldTerminate()) {
            processing_terminated = true;
            break; // 立即退出循环，不进行保存操作
        }

        double timestamp = img_data.first;
        std::string img_path = img_data.second;

        // Store current timestamp and output folder for use in updateExtrinsics
        current_timestamp_ = timestamp;
        output_folder_ = output_folder;

        try {
            // Time interpolation operation
            auto interp_start = std::chrono::high_resolution_clock::now();

            // Interpolate LiDAR pose at this timestamp
            Eigen::Matrix4d lidar_pose;

            if (!interpolatePose(timestamp, trajectory, lidar_pose)) {
                continue;
            }

            if (!use_time) {
                if (!interpolatePose(timestamp + 0.2, trajectory, lidar_pose)) {
                    continue;
                }
            }

            auto interp_end = std::chrono::high_resolution_clock::now();
            double interp_time = std::chrono::duration<double>(interp_end - interp_start).count();
            total_interpolate_time += interp_time;

            // 再次检查终止信号
            if (SignalHandler::getInstance().shouldTerminate()) {
                processing_terminated = true;
                break;
            }

            // Check if there are matching files for calibration update
            if (use_optimize) {
                if (match_files.find(timestamp) != match_files.end() && index_files.find(timestamp) != index_files.end()) {
                    auto update_start = std::chrono::high_resolution_clock::now();
                    if (updateExtrinsics(match_files[timestamp], index_files[timestamp], lidar_pose, cloud)) {
                        extrinsics_update_count++;
                    } else {
                        continue;
                    }
                    auto update_end = std::chrono::high_resolution_clock::now();
                    double update_time = std::chrono::duration<double>(update_end - update_start).count();
                    total_update_extrinsics_time += update_time;
                } else {
                    continue;
                }
            }

            // 再次检查终止信号
            if (SignalHandler::getInstance().shouldTerminate()) {
                processing_terminated = true;
                break;
            }

            // 记录本帧使用的外参（更新后的T_lidar_camera_update_）
            per_frame_extrinsics[timestamp] = T_lidar_camera_update_;

            // 初始化此帧标记
            std::vector<bool> colored_in_this_frame(cloud->points.size(), false);

            // Time colorization operation
            auto color_start = std::chrono::steady_clock::now();

            // Process image frame for colorization with tracking of colored points
            if (!processImageFrame(timestamp, img_path, lidar_pose, cloud,
                                   colored_cloud, point_color_count, output_folder,
                                   colored_in_this_frame)) {
                std::cerr << "Failed to process image frame: " << img_path << std::endl;
                continue;
            }

            auto color_end = std::chrono::steady_clock::now();
            double color_time = std::chrono::duration<double>(color_end - color_start).count();
            total_colorize_time += color_time;
            colorize_count++;

            // Calculate camera pose for visualization
            Eigen::Matrix4d camera_pose = lidar_pose * T_lidar_camera_update_.inverse();

            // Time ROS publishing operations
            auto ros_start = std::chrono::steady_clock::now();

            // Publish to ROS topics if ROS is initialized
            if (ros_initialized_) {
                ros::Time current_ros_time = ros::Time::now();
                publishCameraPose(camera_pose, current_ros_time.toSec());

                // 创建仅包含此帧着色点的点云
                frame_colored_cloud->clear();

                for (size_t i = 0; i < colored_cloud->points.size(); ++i) {
                    if (colored_in_this_frame[i]) {
                        frame_colored_cloud->push_back(colored_cloud->points[i]);
                    }
                }

                if (!frame_colored_cloud->empty()) {
                    publishColoredPointCloud(frame_colored_cloud, current_ros_time.toSec());
                }

                ros::spinOnce(); // 处理ROS回调，但不阻塞
                ros_pub_count++;
            }

            auto ros_end = std::chrono::steady_clock::now();
            if (ros_initialized_) {
                double ros_time = std::chrono::duration<double>(ros_end - ros_start).count();
                total_ros_pub_time += ros_time;
            }

            // 简化的进度显示
            if (img_count % 5 == 0 || img_count == selected_images.size() - 1) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                                           current_time - start_time)
                                           .count();

                float progress = static_cast<float>(img_count + 1) / selected_images.size();
                int elapsed_minutes = static_cast<int>(elapsed_seconds) / 60;
                int elapsed_secs = static_cast<int>(elapsed_seconds) % 60;

                // 简化的进度条
                int bar_width = 30;
                int filled = static_cast<int>(progress * bar_width);

                std::cout << "\rProgress: [";
                for (int i = 0; i < bar_width; ++i) {
                    std::cout << (i < filled ? "=" : " ");
                }
                std::cout << "] " << static_cast<int>(progress * 100) << "% "
                          << "(" << img_count + 1 << "/" << selected_images.size() << ") "
                          << "Time: " << elapsed_minutes << ":" << std::setfill('0') << std::setw(2) << elapsed_secs
                          << "    " << std::flush;
            }

            img_count++;

            // ===== 新增：当前帧渲染并计算PSNR + SSIM =====
            {
                cv::Mat raw_img = cv::imread(img_path, cv::IMREAD_COLOR);
                if (!raw_img.empty()) {
                    cv::Mat undist_img_psnr;
                    if (camera_model_ == "fisheye") {
                        cv::fisheye::undistortImage(raw_img, undist_img_psnr, camera_matrix_, dist_coeffs_, newcamera_matrix_);
                    } else {
                        cv::undistort(raw_img, undist_img_psnr, camera_matrix_, dist_coeffs_, newcamera_matrix_);
                    }
                    cv::resize(undist_img_psnr, undist_img_psnr,
                               cv::Size(resizecamera_matrix_.at<float>(0, 2) * 2,
                                        resizecamera_matrix_.at<float>(1, 2) * 2));

                    Eigen::Matrix4d camera_pose = lidar_pose * T_lidar_camera_update_.inverse();
                    Eigen::Matrix4d world_to_camera = camera_pose.inverse();

                    cv::Mat depth_buf(undist_img_psnr.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
                    cv::Mat render_img(undist_img_psnr.size(), CV_8UC3, cv::Scalar(0, 0, 0));

                    for (const auto &pt : colored_cloud->points) {
                        if (pt.r == 0 && pt.g == 0 && pt.b == 0) continue;
                        Eigen::Vector4d pw(pt.x, pt.y, pt.z, 1.0);
                        Eigen::Vector4d pc = world_to_camera * pw;
                        if (pc[2] <= 0) continue;
                        int px = static_cast<int>(resizecamera_matrix_.at<float>(0, 0) * pc[0] / pc[2] + resizecamera_matrix_.at<float>(0, 2));
                        int py = static_cast<int>(resizecamera_matrix_.at<float>(1, 1) * pc[1] / pc[2] + resizecamera_matrix_.at<float>(1, 2));
                        if (px < 0 || px >= render_img.cols || py < 0 || py >= render_img.rows) continue;
                        float &z = depth_buf.at<float>(py, px);
                        float depth = static_cast<float>(pc[2]);
                        if (depth < z) {
                            z = depth;
                            render_img.at<cv::Vec3b>(py, px) = cv::Vec3b(pt.b, pt.g, pt.r);
                        }
                    }

                    cv::Mat hole_mask(render_img.size(), CV_8U, cv::Scalar(0));
                    for (int yy = 0; yy < depth_buf.rows; ++yy) {
                        const float *zb = depth_buf.ptr<float>(yy);
                        uchar *mp = hole_mask.ptr<uchar>(yy);
                        for (int xx = 0; xx < depth_buf.cols; ++xx) {
                            if (zb[xx] == std::numeric_limits<float>::max()) mp[xx] = 255;
                        }
                    }
                    if (cv::countNonZero(hole_mask) > 0) {
                        cv::Mat median_filtered;
                        cv::medianBlur(render_img, median_filtered, 5);
                        median_filtered.copyTo(render_img, hole_mask);
                        // FIX: proper dilate usage
                        cv::Mat dilated;
                        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
                        cv::dilate(hole_mask, dilated, kernel);
                        cv::Mat blurred;
                        cv::bilateralFilter(render_img, blurred, 5, 20.0, 5.0);
                        blurred.copyTo(render_img, dilated);
                    }

                    // 计算PSNR & SSIM（仅有效且两图都非黑像素）
                    double mse = 0.0;
                    size_t valid = 0;
                    double sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0, sum_y2 = 0.0, sum_xy = 0.0;
                    for (int y = 0; y < render_img.rows; ++y) {
                        const float *zb = depth_buf.ptr<float>(y);
                        const cv::Vec3b *refp = undist_img_psnr.ptr<cv::Vec3b>(y);
                        const cv::Vec3b *renp = render_img.ptr<cv::Vec3b>(y);
                        for (int x = 0; x < render_img.cols; ++x) {
                            if (zb[x] == std::numeric_limits<float>::max()) continue; // 未投影
                            const cv::Vec3b &rpx = refp[x];
                            const cv::Vec3b &opx = renp[x];
                            if ((rpx[0] | rpx[1] | rpx[2]) == 0) continue; // 原图黑色
                            if ((opx[0] | opx[1] | opx[2]) == 0) continue; // 渲染仍黑色
                            for (int c = 0; c < 3; ++c) {
                                double diff = double(rpx[c]) - double(opx[c]);
                                mse += diff * diff;
                            }
                            // 灰度用于SSIM
                            double gx = 0.114 * rpx[0] + 0.587 * rpx[1] + 0.299 * rpx[2];
                            double gy = 0.114 * opx[0] + 0.587 * opx[1] + 0.299 * opx[2];
                            sum_x += gx;
                            sum_y += gy;
                            sum_x2 += gx * gx;
                            sum_y2 += gy * gy;
                            sum_xy += gx * gy;
                            valid++;
                        }
                    }
                    double psnr_db = 0.0;
                    double ssim_val = 0.0;
                    if (valid > 0) {
                        mse /= (valid * 3.0);
                        if (mse > 0.0)
                            psnr_db = 10.0 * std::log10((255.0 * 255.0) / mse);
                        else
                            psnr_db = 99.0;
                        double mu_x = sum_x / valid;
                        double mu_y = sum_y / valid;
                        double var_x = sum_x2 / valid - mu_x * mu_x;
                        if (var_x < 0) var_x = 0;
                        double var_y = sum_y2 / valid - mu_y * mu_y;
                        if (var_y < 0) var_y = 0;
                        double cov_xy = sum_xy / valid - mu_x * mu_y;
                        double C1 = (0.01 * 255) * (0.01 * 255);
                        double C2 = (0.03 * 255) * (0.03 * 255);
                        ssim_val = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (var_x + var_y + C2));
                    }

                    std::string render_path = render_dir + "/" + std::to_string(timestamp) + "_render.png";
                    cv::imwrite(render_path, render_img);

                    if (psnr_csv.is_open()) {
                        psnr_csv << std::fixed << std::setprecision(9)
                                 << timestamp << "," << valid << ","
                                 << std::setprecision(6) << psnr_db << "," << ssim_val << std::endl;
                    }
                }
            }
            // ===== 渲染与PSNR+SSIM结束 =====
        } catch (const std::exception &e) {
            std::cerr << "Error processing image " << img_path << ": " << e.what() << std::endl;
            continue;
        }
    }

    // 优化的终止处理
    if (processing_terminated) {
        std::cout << "\n========== GRACEFUL TERMINATION ==========" << std::endl;
        std::cout << "Processing interrupted by user request." << std::endl;
        if (psnr_csv.is_open()) psnr_csv.close();

        // 计算处理时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        int minutes = static_cast<int>(total_time) / 60;
        int seconds = static_cast<int>(total_time) % 60;

        // 计算进度统计
        int processed_count = img_count;
        int total_count = selected_images.size();
        double progress_percentage = (100.0 * processed_count) / total_count;

        std::cout << "Processing Summary:" << std::endl;
        std::cout << "  - Processed: " << processed_count << "/" << total_count
                  << " images (" << std::fixed << std::setprecision(1)
                  << progress_percentage << "%)" << std::endl;
        std::cout << "  - Runtime: " << minutes << "m " << seconds << "s" << std::endl;

        // 快速保存当前状态
        std::cout << "Saving current state..." << std::endl;

        // 使用超时保存避免长时间阻塞
        auto save_start = std::chrono::steady_clock::now();
        std::string progress_info = "INTERRUPTED - Processed " + std::to_string(processed_count) + "/" + std::to_string(total_count) + " images";

        try {
            // 限制保存时间，避免阻塞
            if (saveVisualization(colored_cloud, point_color_count, output_folder, progress_info)) {
                std::cout << "✓ Current progress saved" << std::endl;
            } else {
                std::cout << "✗ Warning: Could not save all progress data" << std::endl;
            }
        } catch (const std::exception &e) {
            std::cout << "✗ Error saving progress: " << e.what() << std::endl;
        }

        // 快速ROS清理
        if (ros_initialized_) {
            try {
                for (int i = 0; i < 2; ++i) { // 只尝试2次
                    ros::spinOnce();
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            } catch (...) {
                // 忽略ROS清理错误
            }
        }

        std::cout << "========== TERMINATION COMPLETE ==========" << std::endl;
        std::cout << "Partial results saved to: " << output_folder << std::endl;
        std::cout << "You can restart to continue processing." << std::endl;
        std::cout << "===========================================" << std::endl;

        // 确保输出立即刷新
        std::cout.flush();
        std::cerr.flush();

        return true; // 返回true表示成功终止
    }

    // 正常完成的处理
    std::cout << std::endl; // 结束进度条显示

    // 记录总处理时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    int total_minutes = static_cast<int>(total_seconds) / 60;
    int remaining_seconds = static_cast<int>(total_seconds) % 60;

    // Output detailed timing statistics
    std::cout << "\n===== Processing Complete =====" << std::endl;
    std::cout << "Processing completed in " << total_minutes << " minutes and "
              << remaining_seconds << " seconds." << std::endl;

    // 最终保存和报告
    std::string progress_info = "Processing complete. Processed " + std::to_string(img_count) + " images out of " + std::to_string(selected_images.size());
    saveVisualization(colored_cloud, point_color_count, output_folder, progress_info);

    // 新增：完成后用最终彩色点云 + 每帧外参渲染，图像更完整
    try {
        std::string final_render_dir = output_folder + "/render_final";
        if (!fs::exists(final_render_dir)) fs::create_directories(final_render_dir);
        std::string csv_path = final_render_dir + "/psnr_final.csv";
        std::ofstream psnr_csv_final(csv_path);
        if (psnr_csv_final.is_open()) {
            psnr_csv_final << "timestamp,valid_pixels,psnr_db,ssim" << std::endl; // 修改最终CSV头
        }

        for (const auto &img_data : selected_images) {
            double ts = img_data.first;
            const std::string &img_path = img_data.second;

            // 读并去畸变图
            cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
            if (img.empty()) continue;
            cv::Mat undist_img;
            if (camera_model_ == "fisheye") {
                cv::fisheye::undistortImage(img, undist_img, camera_matrix_, dist_coeffs_, newcamera_matrix_);
            } else {
                cv::undistort(img, undist_img, camera_matrix_, dist_coeffs_, newcamera_matrix_);
            }
            cv::resize(undist_img, undist_img,
                       cv::Size(resizecamera_matrix_.at<float>(0, 2) * 2,
                                resizecamera_matrix_.at<float>(1, 2) * 2));

            // 插值获得该帧LiDAR位姿
            Eigen::Matrix4d lidar_pose;
            if (!interpolatePose(ts, trajectory, lidar_pose)) continue;

            // 取该帧记录的外参（若无则使用最终外参）
            Eigen::Matrix4d T_lc = T_lidar_camera_update_;
            auto it = per_frame_extrinsics.find(ts);
            if (it != per_frame_extrinsics.end()) T_lc = it->second;

            // 相机位姿
            Eigen::Matrix4d camera_pose = lidar_pose * T_lc.inverse();
            Eigen::Matrix4d world_to_camera = camera_pose.inverse();

            // 渲染缓冲
            cv::Mat depth_buf(undist_img.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
            cv::Mat render_img(undist_img.size(), CV_8UC3, cv::Scalar(0, 0, 0));

            // Z-buffer投影
            for (const auto &pt : colored_cloud->points) {
                if (pt.r == 0 && pt.g == 0 && pt.b == 0) continue;
                Eigen::Vector4d pw(pt.x, pt.y, pt.z, 1.0);
                Eigen::Vector4d pc = world_to_camera * pw;
                if (pc[2] <= 0) continue;
                int px = static_cast<int>(resizecamera_matrix_.at<float>(0, 0) * pc[0] / pc[2] + resizecamera_matrix_.at<float>(0, 2));
                int py = static_cast<int>(resizecamera_matrix_.at<float>(1, 1) * pc[1] / pc[2] + resizecamera_matrix_.at<float>(1, 2));
                if (px < 0 || px >= render_img.cols || py < 0 || py >= render_img.rows) continue;
                float &z = depth_buf.at<float>(py, px);
                float depth = static_cast<float>(pc[2]);
                if (depth < z) {
                    z = depth;
                    render_img.at<cv::Vec3b>(py, px) = cv::Vec3b(pt.b, pt.g, pt.r);
                }
            }

            // 按fillHoles思想做中值填孔 + 局部平滑
            cv::Mat hole_mask(render_img.size(), CV_8U, cv::Scalar(0));
            for (int yy = 0; yy < depth_buf.rows; ++yy) {
                const float *zb = depth_buf.ptr<float>(yy);
                uchar *mp = hole_mask.ptr<uchar>(yy);
                for (int xx = 0; xx < depth_buf.cols; ++xx) {
                    if (zb[xx] == std::numeric_limits<float>::max()) mp[xx] = 255;
                }
            }
            if (cv::countNonZero(hole_mask) > 0) {
                cv::Mat median_filtered;
                cv::medianBlur(render_img, median_filtered, 5);
                median_filtered.copyTo(render_img, hole_mask);
                // FIX: proper dilate usage
                cv::Mat dilated;
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
                cv::dilate(hole_mask, dilated, kernel);
                cv::Mat blurred;
                cv::bilateralFilter(render_img, blurred, 5, 20.0, 5.0);
                blurred.copyTo(render_img, dilated);
            }

            // 计算PSNR & SSIM（仅有效投影且两图非黑像素）
            double mse = 0.0;
            size_t valid = 0;
            double sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0, sum_y2 = 0.0, sum_xy = 0.0;
            for (int y = 0; y < render_img.rows; ++y) {
                const float *zb = depth_buf.ptr<float>(y);
                const cv::Vec3b *refp = undist_img.ptr<cv::Vec3b>(y);
                const cv::Vec3b *renp = render_img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < render_img.cols; ++x) {
                    if (zb[x] == std::numeric_limits<float>::max()) continue;
                    const cv::Vec3b &rpx = refp[x];
                    const cv::Vec3b &opx = renp[x];
                    if ((rpx[0] | rpx[1] | rpx[2]) == 0) continue;
                    if ((opx[0] | opx[1] | opx[2]) == 0) continue;
                    for (int c = 0; c < 3; ++c) {
                        double diff = double(rpx[c]) - double(opx[c]);
                        mse += diff * diff;
                    }
                    double gx = 0.114 * rpx[0] + 0.587 * rpx[1] + 0.299 * rpx[2];
                    double gy = 0.114 * opx[0] + 0.587 * opx[1] + 0.299 * opx[2];
                    sum_x += gx;
                    sum_y += gy;
                    sum_x2 += gx * gx;
                    sum_y2 += gy * gy;
                    sum_xy += gx * gy;
                    valid++;
                }
            }
            double psnr_db = 0.0;
            double ssim_val = 0.0;
            if (valid > 0) {
                mse /= (valid * 3.0);
                psnr_db = (mse > 0.0) ? 10.0 * std::log10((255.0 * 255.0) / mse) : 99.0;
                double mu_x = sum_x / valid;
                double mu_y = sum_y / valid;
                double var_x = sum_x2 / valid - mu_x * mu_x;
                if (var_x < 0) var_x = 0;
                double var_y = sum_y2 / valid - mu_y * mu_y;
                if (var_y < 0) var_y = 0;
                double cov_xy = sum_xy / valid - mu_x * mu_y;
                double C1 = (0.01 * 255) * (0.01 * 255);
                double C2 = (0.03 * 255) * (0.03 * 255);
                ssim_val = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (var_x + var_y + C2));
            }

            // 保存渲染图与记录CSV
            std::string render_path = final_render_dir + "/" + std::to_string(ts) + "_render.png";
            cv::imwrite(render_path, render_img);
            if (psnr_csv_final.is_open()) {
                psnr_csv_final << std::fixed << std::setprecision(9) << ts << "," << valid << ","
                               << std::setprecision(6) << psnr_db << "," << ssim_val << std::endl;
            }
        }
        if (psnr_csv_final.is_open()) psnr_csv_final.close();
        std::cout << "Final render images saved to: " << final_render_dir << ", PSNR CSV: " << csv_path << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error during final rendering: " << e.what() << std::endl;
    }

    const std::string &camera_model = Config::getInstance().cameraParams().camera_model;
    const auto &proj_params2 = Config::getInstance().projectionParams();

    // Clear the stored pointers as we're done
    if (output_cloud_ptr) {
        *output_cloud_ptr = nullptr;
    }
    if (output_color_count_ptr) {
        *output_color_count_ptr = nullptr;
    }

    // Generate and save additional outputs
    saveOptimizedCameraPoses(trajectory, output_folder);

    return true;
}

bool CalibProcessor::preprocess(const std::string &image_folder,
                                const std::string &trajectory_file,
                                const std::string &pcd_file,
                                const std::string &output_folder) {
    // 创建输出目录（如果不存在）
    if (output_folder.empty()) {
        std::cerr << "Error: Output folder path is empty" << std::endl;
        return false;
    }

    try {
        if (!fs::exists(output_folder)) {
            if (!fs::create_directories(output_folder)) {
                std::cerr << "Failed to create output directory: " << output_folder
                          << std::endl;
                return false;
            }
        }
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return false;
    }

    // 创建所有必需的子目录
    std::string undist_dir = output_folder + "/undist_img";
    std::string rimg_dir = output_folder + "/rimg";
    std::string index_dir = output_folder + "/index_maps";

    try {
        if (!fs::exists(undist_dir))
            fs::create_directories(undist_dir);
        if (!fs::exists(rimg_dir))
            fs::create_directories(rimg_dir);
        if (!fs::exists(index_dir))
            fs::create_directories(index_dir);
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Failed to create subdirectories: " << e.what() << std::endl;
        return false;
    }

    std::cout << "Loading trajectory data..." << std::endl;
    auto load_start = std::chrono::high_resolution_clock::now();

    // 加载轨迹数据
    std::vector<std::pair<double, Eigen::Matrix4d>> trajectory;
    std::ifstream traj_file(trajectory_file);
    if (!traj_file.is_open()) {
        std::cerr << "Failed to open trajectory file: " << trajectory_file
                  << std::endl;
        return false;
    }

    // 解析轨迹文件（time, x, y, z, qx, qy, qz, qw）
    std::string line;
    while (std::getline(traj_file, line)) {
        std::istringstream iss(line);
        double time, x, y, z, qx, qy, qz, qw;
        if (!(iss >> time >> x >> y >> z >> qx >> qy >> qz >> qw)) {
            continue;
        }

        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        Eigen::Quaterniond q(qw, qx, qy, qz);
        pose.block<3, 3>(0, 0) = q.toRotationMatrix();
        pose.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);

        trajectory.push_back({time, pose});
    }

    // 按时间戳排序轨迹
    std::sort(trajectory.begin(), trajectory.end(), compareTimestamps);

    auto traj_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - load_start)
                         .count();
    std::cout << "Loaded " << trajectory.size() << " trajectory points in "
              << traj_time / 1000.0 << "s" << std::endl;

    // 加载点云 (不进行下采样)
    std::cout << "Loading point cloud..." << std::endl;
    auto cloud_start = std::chrono::high_resolution_clock::now();

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *cloud) == -1) {
        std::cerr << "Failed to load point cloud: " << pcd_file << std::endl;
        return false;
    }

    auto cloud_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::high_resolution_clock::now() - cloud_start)
                          .count();
    std::cout << "Loaded point cloud with " << cloud->points.size()
              << " points in " << cloud_time / 1000.0 << "s" << std::endl;

    // 获取所有图像文件
    std::cout << "Scanning image directory..." << std::endl;
    auto img_scan_start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<double, std::string>> image_files;

    try {
        for (const auto &entry : fs::directory_iterator(image_folder)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                // 假设文件名是时间戳
                try {
                    double timestamp =
                        std::stod(filename.substr(0, filename.find_last_of('.')));
                    image_files.push_back({timestamp, entry.path().string()});
                } catch (const std::exception &e) {
                    std::cerr << "Warning: Could not parse timestamp from filename: "
                              << filename << std::endl;
                }
            }
        }
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Error scanning image directory: " << e.what() << std::endl;
        return false;
    }

    // 按时间戳排序图像
    std::sort(image_files.begin(), image_files.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    auto img_scan_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - img_scan_start)
            .count();
    std::cout << "Found and sorted " << image_files.size() << " images in "
              << img_scan_time / 1000.0 << "s" << std::endl;

    // 使用配置中的采样步长
    const Config &config = Config::getInstance();
    int sampling_step = config.processingParams().img_sampling_step;

    // 采样图像以提高处理速度
    std::vector<std::pair<double, std::string>> selected_images;
    for (size_t i = 0; i < image_files.size(); i += sampling_step) {
        selected_images.push_back(image_files[i]);
    }

    std::cout << "Selected " << selected_images.size() << " images out of "
              << image_files.size() << " for processing (sampling every "
              << sampling_step << " images)" << std::endl;

    // 设置线程数为4
    const size_t num_threads = 8;
    std::cout << "Using " << num_threads << " threads for parallel processing"
              << std::endl;

    // 用于线程同步的变量
    std::atomic<int> processed_count(0);
    std::atomic<bool> should_terminate(false);
    std::mutex cout_mutex;

    // 线程完成标志
    std::vector<bool> thread_finished(num_threads, false);

    // 确定每个线程的批处理大小
    size_t batch_size = (selected_images.size() + num_threads - 1) / num_threads;

    // 并行处理图像
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;

    // 启动线程
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            // 确定该线程的处理范围
            size_t start_idx = t * batch_size;
            size_t end_idx = std::min(start_idx + batch_size, selected_images.size());

            // 线程本地的ImageProcess对象
            ImageProcess imgProc;

            // 处理分配给该线程的每张图像
            for (size_t i = start_idx; i < end_idx && !should_terminate; ++i) {
                // 检查是否应该终止
                if (SignalHandler::getInstance().shouldTerminate()) {
                    should_terminate = true;
                    break;
                }

                const auto &img_data = selected_images[i];
                double timestamp = img_data.first;
                std::string img_path = img_data.second;

                try {
                    // 加载图像
                    cv::Mat img = cv::imread(img_path);
                    if (img.empty()) {
                        {
                            std::lock_guard<std::mutex> lock(cout_mutex);
                            std::cerr << "Failed to load image: " << img_path << std::endl;
                        }
                        continue;
                    }

                    // 检查时间戳是否在轨迹范围内
                    if (timestamp <= trajectory.front().first || timestamp >= trajectory.back().first) {
                        continue;
                    }

                    // 查找包含时间戳的轨迹点
                    size_t idx = 0;
                    while (idx < trajectory.size() - 1 && trajectory[idx + 1].first < timestamp) {
                        idx++;
                    }

                    // 插值计算位姿
                    double t1 = trajectory[idx].first;
                    double t2 = trajectory[idx + 1].first;
                    Eigen::Matrix4d pose1 = trajectory[idx].second;
                    Eigen::Matrix4d pose2 = trajectory[idx + 1].second;

                    double alpha = (timestamp - t1) / (t2 - t1);

                    // 位置插值
                    Eigen::Vector3d pos1 = pose1.block<3, 1>(0, 3);
                    Eigen::Vector3d pos2 = pose2.block<3, 1>(0, 3);
                    Eigen::Vector3d pos = pos1 + alpha * (pos2 - pos1);

                    // 旋转插值
                    Eigen::Quaterniond q1(pose1.block<3, 3>(0, 0));
                    Eigen::Quaterniond q2(pose2.block<3, 3>(0, 0));
                    Eigen::Quaterniond q = q1.slerp(alpha, q2);

                    Eigen::Matrix4d lidar_pose = Eigen::Matrix4d::Identity();
                    lidar_pose.block<3, 3>(0, 0) = q.toRotationMatrix();
                    lidar_pose.block<3, 1>(0, 3) = pos;

                    // 计算相机位姿
                    Eigen::Matrix4d camera_pose = lidar_pose * T_lidar_camera_.inverse();

                    // 获取相机模型
                    const std::string &camera_model =
                        Config::getInstance().cameraParams().camera_model;

                    // 根据相机模型去除图像畸变
                    cv::Mat undistorted_img;
                    if (camera_model == "fisheye") {
                        cv::fisheye::undistortImage(img, undistorted_img, camera_matrix_,
                                                    dist_coeffs_, newcamera_matrix_);
                    } else if (camera_model == "pinhole") {
                        cv::undistort(img, undistorted_img, camera_matrix_, dist_coeffs_,
                                      newcamera_matrix_);
                    }

                    // 调整图像大小
                    cv::Mat resized_img;
                    cv::resize(undistorted_img, resized_img,
                               cv::Size(resizecamera_matrix_.at<float>(0, 2) * 2,
                                        resizecamera_matrix_.at<float>(1, 2) * 2),
                               0, 0, cv::INTER_LINEAR);

                    // 保存无畸变图像
                    std::string undist_img_path =
                        undist_dir + "/" + std::to_string(timestamp) + ".png";
                    cv::imwrite(undist_img_path, resized_img);

                    // 转换点云 - 线程本地副本
                    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(
                        new pcl::PointCloud<pcl::PointXYZI>);
                    transformed_cloud->reserve(cloud->points.size() / 2); // 预分配足够的空间

                    // 存储原始索引到转换后点云的映射
                    std::vector<int> index_mapping;
                    index_mapping.reserve(cloud->points.size() / 2);

                    // 转换点并过滤掉相机后方的点
                    Eigen::Matrix4d cam_inv_transform = camera_pose.inverse();
                    for (size_t p_idx = 0; p_idx < cloud->points.size(); ++p_idx) {
                        const auto &point = cloud->points[p_idx];
                        Eigen::Vector4d pt_world(point.x, point.y, point.z, 1.0);
                        Eigen::Vector4d pt_camera = cam_inv_transform * pt_world;

                        // 只添加相机前方的点
                        if (pt_camera[2] > 0) {
                            pcl::PointXYZI transformed_point;
                            transformed_point.x = pt_camera[0];
                            transformed_point.y = pt_camera[1];
                            transformed_point.z = pt_camera[2];
                            transformed_point.intensity = point.intensity;

                            transformed_cloud->push_back(transformed_point);
                            index_mapping.push_back(p_idx);
                        }
                    }

                    // 投影点云并获取索引映射
                    auto result =
                        imgProc.projectPinholeWithIndices(transformed_cloud, true);
                    cv::Mat projected_image = result.first;
                    cv::Mat index_map = result.second;

                    // 将投影索引映射回原始点云索引
                    cv::Mat original_index_map(index_map.size(), CV_32S, cv::Scalar(-1));
                    for (int y = 0; y < index_map.rows; y++) {
                        for (int x = 0; x < index_map.cols; x++) {
                            int idx = index_map.at<int>(y, x);
                            if (idx >= 0 && idx < index_mapping.size()) {
                                original_index_map.at<int>(y, x) = index_mapping[idx];
                            }
                        }
                    }

                    // 保存投影图像
                    std::string projected_img_path =
                        rimg_dir + "/" + std::to_string(timestamp) + ".png";
                    cv::imwrite(projected_img_path, projected_image);

                    // 保存索引映射为可视化图像 (使用原始索引映射)
                    // 查找最大索引值用于颜色映射
                    int max_index = 0;
                    for (int y = 0; y < original_index_map.rows; y++) {
                        for (int x = 0; x < original_index_map.cols; x++) {
                            int idx = original_index_map.at<int>(y, x);
                            if (idx > max_index)
                                max_index = idx;
                        }
                    }

                    // 创建索引图像
                    cv::Mat index_image(original_index_map.size(), CV_16UC3);
                    for (int y = 0; y < original_index_map.rows; y++) {
                        for (int x = 0; x < original_index_map.cols; x++) {
                            int idx = original_index_map.at<int>(y, x);
                            if (idx >= 0) { // 有效点索引
                                // 编码索引为3个16位通道
                                index_image.at<cv::Vec3w>(y, x)[0] = idx & 0xFFFF;
                                index_image.at<cv::Vec3w>(y, x)[1] = (idx >> 16) & 0xFFFF;
                                index_image.at<cv::Vec3w>(y, x)[2] = 0;
                            } else {
                                // 无点映射到此像素
                                index_image.at<cv::Vec3w>(y, x) = cv::Vec3w(0, 0, 0);
                            }
                        }
                    }

                    // 保存索引图像
                    std::string index_map_path =
                        index_dir + "/" + std::to_string(timestamp) + "_index.png";
                    cv::imwrite(index_map_path, index_image);

                    // 创建可视化图像
                    cv::Mat vis_index_map(original_index_map.size(), CV_8UC3,
                                          cv::Scalar(0, 0, 0));
                    for (int y = 0; y < original_index_map.rows; y++) {
                        for (int x = 0; x < original_index_map.cols; x++) {
                            int idx = original_index_map.at<int>(y, x);
                            if (idx >= 0) { // 有效点索引
                                // 创建基于索引的颜色（仅用于可视化）
                                float normalized_idx = static_cast<float>(idx) / max_index;
                                int hue = static_cast<int>(normalized_idx * 180);

                                // 将HSV转为RGB用于可视化（完全饱和，完全亮度）
                                cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
                                cv::Mat rgb;
                                cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);

                                vis_index_map.at<cv::Vec3b>(y, x) = rgb.at<cv::Vec3b>(0, 0);
                            }
                        }
                    }

                    // 保存可视化图像
                    std::string vis_index_path =
                        index_dir + "/" + std::to_string(timestamp) + "_index_vis.png";
                    cv::imwrite(vis_index_path, vis_index_map);

                    // 保存二进制索引文件 - 增强的安全版本
                    std::string binary_index_path =
                        index_dir + "/" + std::to_string(timestamp) + "_index.bin";
                    std::ofstream index_file(binary_index_path, std::ios::binary);
                    if (index_file.is_open()) {
                        int rows = original_index_map.rows;
                        int cols = original_index_map.cols;

                        // 确保矩阵在内存中连续
                        cv::Mat continuous_map;
                        if (original_index_map.isContinuous()) {
                            continuous_map = original_index_map;
                        } else {
                            continuous_map = original_index_map.clone();
                        }

                        // 写入维度信息，检查写入是否成功
                        bool write_success = true;
                        index_file.write(reinterpret_cast<const char *>(&rows),
                                         sizeof(int));
                        write_success &= !index_file.fail();

                        index_file.write(reinterpret_cast<const char *>(&cols),
                                         sizeof(int));
                        write_success &= !index_file.fail();

                        // 分块写入数据以避免大量内存操作
                        const int chunk_rows = 64;
                        for (int row = 0; row < rows && write_success; row += chunk_rows) {
                            int current_rows = std::min(chunk_rows, rows - row);
                            size_t data_size = current_rows * cols * sizeof(int);

                            index_file.write(
                                reinterpret_cast<const char *>(continuous_map.ptr(row)),
                                data_size);
                            write_success &= !index_file.fail();

                            // 确保写入后刷新缓冲区
                            index_file.flush();
                        }

                        index_file.close();

                        if (!write_success) {
                            std::lock_guard<std::mutex> lock(cout_mutex);
                            std::cerr << "Error writing index map to: " << binary_index_path
                                      << std::endl;
                        }
                    } else {
                        std::lock_guard<std::mutex> lock(cout_mutex);
                        std::cerr << "Failed to open index file for writing: "
                                  << binary_index_path << std::endl;
                    }

                    // 显示处理进度
                    int curr_count = ++processed_count;
                    if (curr_count % 10 == 0 || curr_count == selected_images.size()) {
                        std::lock_guard<std::mutex> lock(cout_mutex);
                        auto current_time = std::chrono::high_resolution_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                           current_time - start_time)
                                           .count();
                        double progress = 100.0 * curr_count / selected_images.size();

                        // 绘制进度条
                        int bar_width = 50;
                        int filled_width = static_cast<int>(bar_width * progress / 100.0);

                        std::cout << "Processing: [";
                        for (int i = 0; i < bar_width; ++i) {
                            if (i < filled_width)
                                std::cout << "=";
                            else
                                std::cout << " ";
                        }
                        std::cout << "] " << std::fixed << std::setprecision(1) << progress
                                  << "% " << curr_count << "/" << selected_images.size();

                        if (elapsed > 0 && curr_count > 10) {
                            double images_per_second =
                                static_cast<double>(curr_count) / elapsed;
                            double remaining_seconds =
                                (selected_images.size() - curr_count) / images_per_second;

                            int minutes = static_cast<int>(remaining_seconds) / 60;
                            int seconds = static_cast<int>(remaining_seconds) % 60;

                            std::cout << " | " << std::fixed << std::setprecision(2)
                                      << images_per_second << " img/s"
                                      << " | ETA: " << minutes << "m " << seconds << "s";
                        }
                        std::cout << std::endl;
                    }

                } catch (const std::exception &e) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cerr << "Error processing image " << img_path << ": " << e.what()
                              << std::endl;
                }
            }

            thread_finished[t] = true;
        });
    }

    // 等待所有线程完成或处理终止请求
    while (true) {
        if (SignalHandler::getInstance().shouldTerminate() && !should_terminate) {
            should_terminate = true;
            std::cout << "\nTermination requested. Waiting for threads to finish..."
                      << std::endl;
        }

        // 检查是否所有线程都已完成
        bool all_done = true;
        for (size_t t = 0; t < num_threads; ++t) {
            if (!thread_finished[t]) {
                all_done = false;
                break;
            }
        }

        if (all_done)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 等待所有线程加入
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // 计算总处理时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time =
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
            .count();
    int minutes = static_cast<int>(total_time) / 60;
    int seconds = static_cast<int>(total_time) % 60;

    if (should_terminate) {
        std::cout << "\nPreprocessing interrupted! Processed " << processed_count
                  << " images in " << minutes << " minutes and " << seconds
                  << " seconds." << std::endl;
    } else {
        std::cout << "\nPreprocessing complete! Processed " << processed_count
                  << " images in " << minutes << " minutes and " << seconds
                  << " seconds." << std::endl;
    }

    return true;
}

bool CalibProcessor::initialize(const std::string &config_file) {
    // Load configuration
    Config &config = Config::getInstance();
    if (!config.loadFromYAML(config_file)) {
        std::cerr << "Failed to load config from: " << config_file << std::endl;
        return false;
    }

    // Store parameters for later use
    image_folder_ = config.processingParams().img_path;
    trajectory_file_ = config.processingParams().traj_path;
    pcd_file_ = config.processingParams().pcd_path;
    output_folder_ = config.processingParams().output_path;
    config_file_ = config_file;

    // Load calibration parameters
    if (!loadCalibrationParameters(config_file)) {
        std::cerr << "Failed to load calibration parameters" << std::endl;
        return false;
    }

    // Display loaded parameters for confirmation
    std::cout << "Successfully loaded configuration:" << std::endl;
    std::cout << "  - Image path: " << image_folder_ << std::endl;
    std::cout << "  - Trajectory path: " << trajectory_file_ << std::endl;
    std::cout << "  - Point cloud path: " << pcd_file_ << std::endl;
    std::cout << "  - Output path: " << output_folder_ << std::endl;

    return true;
}

bool CalibProcessor::run() {
    std::cout << "Starting processing with parameters:" << std::endl;

    // Validate input paths
    if (image_folder_.empty()) {
        std::cerr << "Error: Image folder path is empty" << std::endl;
        return false;
    }

    if (trajectory_file_.empty()) {
        std::cerr << "Error: Trajectory file path is empty" << std::endl;
        return false;
    }

    if (pcd_file_.empty()) {
        std::cerr << "Error: Point cloud file path is empty" << std::endl;
        return false;
    }

    if (output_folder_.empty()) {
        std::cerr << "Error: Output folder path is empty" << std::endl;
        return false;
    }

    std::cout << "Image folder: " << image_folder_ << std::endl;
    std::cout << "Trajectory file: " << trajectory_file_ << std::endl;
    std::cout << "Point cloud file: " << pcd_file_ << std::endl;
    std::cout << "Output folder: " << output_folder_ << std::endl;

    // Initialize signal handler to enable clean termination on Ctrl+C
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> output_cloud = nullptr;
    std::vector<int> *output_color_count = nullptr;

    SignalHandler::getInstance().init();

    // Run the main processing
    return processImagesAndPointCloud(image_folder_, trajectory_file_, pcd_file_,
                                      output_folder_, &output_cloud,
                                      &output_color_count);
}

bool CalibProcessor::runPreprocessing() {
    std::cout << "Starting preprocessing with parameters:" << std::endl;

    // Validate input paths
    if (image_folder_.empty()) {
        std::cerr << "Error: Image folder path is empty" << std::endl;
        return false;
    }

    if (trajectory_file_.empty()) {
        std::cerr << "Error: Trajectory file path is empty" << std::endl;
        return false;
    }

    if (pcd_file_.empty()) {
        std::cerr << "Error: Point cloud file path is empty" << std::endl;
        return false;
    }

    if (output_folder_.empty()) {
        std::cerr << "Error: Output folder path is empty" << std::endl;
        return false;
    }

    std::cout << "Image folder: " << image_folder_ << std::endl;
    std::cout << "Trajectory file: " << trajectory_file_ << std::endl;
    std::cout << "Point cloud file: " << pcd_file_ << std::endl;
    std::cout << "Output folder: " << output_folder_ << std::endl;

    // Initialize signal handler to enable clean termination on Ctrl+C
    SignalHandler::getInstance().init();

    // Run preprocessing
    return preprocess(image_folder_, trajectory_file_, pcd_file_, output_folder_);
}

bool CalibProcessor::saveOptimizedCameraPoses(
    const std::vector<std::pair<double, Eigen::Matrix4d>> &trajectory,
    const std::string &output_folder) {
    // Create directory for trajectory if it doesn't exist
    std::string traj_dir = output_folder + "/trajectory";
    if (!fs::exists(traj_dir)) {
        fs::create_directories(traj_dir);
    }

    // Save camera poses to file
    std::string pose_file_path = traj_dir + "/optimized_camera_poses.txt";
    std::ofstream pose_file(pose_file_path);
    if (!pose_file.is_open()) {
        std::cerr << "Failed to create optimized camera pose file: "
                  << pose_file_path << std::endl;
        return false;
    }

    for (const auto &pose_data : trajectory) {
        double timestamp = pose_data.first;
        // Calculate camera pose from LiDAR pose
        Eigen::Matrix4d camera_pose =
            pose_data.second * T_lidar_camera_update_.inverse();

        // Extract position
        Eigen::Vector3d position = camera_pose.block<3, 1>(0, 3);

        // Extract rotation as quaternion
        Eigen::Matrix3d rot = camera_pose.block<3, 3>(0, 0);
        Eigen::Quaterniond q(rot);
        q.normalize();

        // Write timestamp, position, and quaternion to file
        pose_file << std::fixed << std::setprecision(9) << timestamp << " "
                  << std::setprecision(6) << position.x() << " " << position.y()
                  << " " << position.z() << " " << q.x() << " " << q.y() << " "
                  << q.z() << " " << q.w() << std::endl;
    }

    pose_file.close();
    std::cout << "Saved optimized camera poses to: " << pose_file_path
              << std::endl;
    return true;
}

bool CalibProcessor::saveOptimizedExtrinsics(
    const std::vector<std::pair<double, Eigen::Matrix4d>> &trajectory,
    const std::string &output_folder) {
    // Create directory for calibration if it doesn't exist
    std::string calib_dir = output_folder + "/calibration";
    if (!fs::exists(calib_dir)) {
        fs::create_directories(calib_dir);
    }

    // Save extrinsics to file
    std::string extrinsics_file_path = calib_dir + "/optimized_extrinsics.txt";
    std::ofstream extr_file(extrinsics_file_path);
    if (!extr_file.is_open()) {
        std::cerr << "Failed to create optimized extrinsics file: "
                  << extrinsics_file_path << std::endl;
        return false;
    }

    // For each timestamp, save the optimized extrinsics
    // We're using the same T_lidar_camera_update_ for all frames in this
    // implementation
    for (const auto &pose_data : trajectory) {
        double timestamp = pose_data.first;

        // Extract translation from extrinsics
        Eigen::Vector3d translation = T_lidar_camera_update_.block<3, 1>(0, 3);

        // Extract rotation as quaternion
        Eigen::Matrix3d rot = T_lidar_camera_update_.block<3, 3>(0, 0);
        Eigen::Quaterniond q(rot);
        q.normalize();

        // Write timestamp, position, and quaternion to file
        extr_file << std::fixed << std::setprecision(9) << timestamp << " "
                  << std::setprecision(6) << translation.x() << " "
                  << translation.y() << " " << translation.z() << " " << q.x()
                  << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }

    extr_file.close();
    std::cout << "Saved optimized extrinsics to: " << extrinsics_file_path
              << std::endl;
    return true;
}

bool CalibProcessor::createCameraTrajectoryVisualization(
    const std::vector<std::pair<double, Eigen::Matrix4d>> &trajectory,
    const std::string &output_folder) {
    // Create directory for trajectory if it doesn't exist
    std::string traj_dir = output_folder + "/trajectory";
    if (!fs::exists(traj_dir)) {
        fs::create_directories(traj_dir);
    }

    // Create a new point cloud with camera models at each pose
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr traj_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    // Define camera model vertices (simplified camera frustum)
    // Origin at camera center
    Eigen::Vector3d cam_center(0, 0, 0);
    // Camera frustum corners (representing field of view)
    double size = 0.1; // Size of the camera model
    Eigen::Vector3d top_left(-size, size, size * 2);
    Eigen::Vector3d top_right(size, size, size * 2);
    Eigen::Vector3d bottom_left(-size, -size, size * 2);
    Eigen::Vector3d bottom_right(size, -size, size * 2);

    // Define colors (RGB)
    uint8_t traj_r = 255, traj_g = 0, traj_b = 0; // Red for trajectory points
    uint8_t cam_r = 0, cam_g = 255, cam_b = 0;    // Green for camera model

    // Process each trajectory point
    for (const auto &pose_data : trajectory) {
        // Calculate camera pose from LiDAR pose
        Eigen::Matrix4d camera_pose =
            pose_data.second * T_lidar_camera_update_.inverse();

        // Extract position
        Eigen::Vector3d position = camera_pose.block<3, 1>(0, 3);

        // Extract rotation
        Eigen::Matrix3d rotation = camera_pose.block<3, 3>(0, 0);

        // Add trajectory point (camera center) in red
        pcl::PointXYZRGB traj_pt;
        traj_pt.x = position.x();
        traj_pt.y = position.y();
        traj_pt.z = position.z();
        traj_pt.r = traj_r;
        traj_pt.g = traj_g;
        traj_pt.b = traj_b;
        traj_cloud->points.push_back(traj_pt);

        // Transform camera model vertices to world coordinates
        Eigen::Vector3d world_center = position;
        Eigen::Vector3d world_top_left = position + rotation * top_left;
        Eigen::Vector3d world_top_right = position + rotation * top_right;
        Eigen::Vector3d world_bottom_left = position + rotation * bottom_left;
        Eigen::Vector3d world_bottom_right = position + rotation * bottom_right;

        // Add camera center
        pcl::PointXYZRGB cam_pt_center;
        cam_pt_center.x = world_center.x();
        cam_pt_center.y = world_center.y();
        cam_pt_center.z = world_center.z();
        cam_pt_center.r = cam_r;
        cam_pt_center.g = cam_g;
        cam_pt_center.b = cam_b;
        traj_cloud->points.push_back(cam_pt_center);

        // Add camera frustum corners
        pcl::PointXYZRGB cam_pt_tl, cam_pt_tr, cam_pt_bl, cam_pt_br;

        cam_pt_tl.x = world_top_left.x();
        cam_pt_tl.y = world_top_left.y();
        cam_pt_tl.z = world_top_left.z();
        cam_pt_tl.r = cam_r;
        cam_pt_tl.g = cam_g;
        cam_pt_tl.b = cam_b;
        traj_cloud->points.push_back(cam_pt_tl);

        cam_pt_tr.x = world_top_right.x();
        cam_pt_tr.y = world_top_right.y();
        cam_pt_tr.z = world_top_right.z();
        cam_pt_tr.r = cam_r;
        cam_pt_tr.g = cam_g;
        cam_pt_tr.b = cam_b;
        traj_cloud->points.push_back(cam_pt_tr);

        cam_pt_bl.x = world_bottom_left.x();
        cam_pt_bl.y = world_bottom_left.y();
        cam_pt_bl.z = world_bottom_left.z();
        cam_pt_bl.r = cam_r;
        cam_pt_bl.g = cam_g;
        cam_pt_bl.b = cam_b;
        traj_cloud->points.push_back(cam_pt_bl);

        cam_pt_br.x = world_bottom_right.x();
        cam_pt_br.y = world_bottom_right.y();
        cam_pt_br.z = world_bottom_right.z();
        cam_pt_br.r = cam_r;
        cam_pt_br.g = cam_g;
        cam_pt_br.b = cam_b;
        traj_cloud->points.push_back(cam_pt_br);

        // Add lines connecting corners to center to create camera frustum
        // We do this by adding pairs of points to create line segments
        // First, duplicate the center point multiple times
        for (int i = 0; i < 4; i++) {
            pcl::PointXYZRGB center_dup = cam_pt_center;
            traj_cloud->points.push_back(center_dup);
        }

        // Then add the four corners again to create lines (as point pairs)
        traj_cloud->points.push_back(cam_pt_tl);
        traj_cloud->points.push_back(cam_pt_tr);
        traj_cloud->points.push_back(cam_pt_tr); // Fixed: cam_pt_pt_tr -> cam_pt_tr
        traj_cloud->points.push_back(cam_pt_br);
        traj_cloud->points.push_back(cam_pt_br);
        traj_cloud->points.push_back(cam_pt_bl);
        traj_cloud->points.push_back(cam_pt_bl);
        traj_cloud->points.push_back(cam_pt_tl);
    }

    // Save the point cloud
    std::string traj_viz_path = traj_dir + "/camera_trajectory_with_model.pcd";
    pcl::io::savePCDFileBinary(traj_viz_path, *traj_cloud);

    std::cout << "Saved camera trajectory visualization with camera models to: "
              << traj_viz_path << std::endl;
    std::cout << "Open with PCL viewer to see lines connecting points (e.g., "
                 "pcl_viewer -ax 1 "
              << traj_viz_path << ")" << std::endl;

    return true;
}

void CalibProcessor::publishColoredPointCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, double timestamp) {
    if (!ros_initialized_ || !nh_) {
        return;
    }

    // Convert PCL point cloud to ROS message
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud,
                  cloud_msg); // This will now work with the proper include

    // Set header information with current time
    cloud_msg.header.stamp = ros::Time(timestamp);
    cloud_msg.header.frame_id = frame_id_;

    // Publish the point cloud
    pointcloud_pub_.publish(cloud_msg);
}

void CalibProcessor::publishCameraPose(const Eigen::Matrix4d &camera_pose,
                                       double timestamp) {
    if (!ros_initialized_ || !nh_) {
        return;
    }

    // Extract position
    Eigen::Vector3d position = camera_pose.block<3, 1>(0, 3);

    // Extract rotation as quaternion
    Eigen::Matrix3d rot = camera_pose.block<3, 3>(0, 0);
    Eigen::Quaterniond q(rot);
    q.normalize();

    ros::Time ros_time = ros::Time(timestamp);

    // Create a transform message for tf broadcasting
    geometry_msgs::TransformStamped transform_stamped;
    transform_stamped.header.stamp = ros_time;
    transform_stamped.header.frame_id = frame_id_;
    transform_stamped.child_frame_id = "camera";

    // Set the translation
    transform_stamped.transform.translation.x = position.x();
    transform_stamped.transform.translation.y = position.y();
    transform_stamped.transform.translation.z = position.z();

    // Set the rotation
    transform_stamped.transform.rotation.x = q.x();
    transform_stamped.transform.rotation.y = q.y();
    transform_stamped.transform.rotation.z = q.z();
    transform_stamped.transform.rotation.w = q.w();

    // Broadcast the transform
    tf_broadcaster_.sendTransform(transform_stamped);

    // Create marker array for camera visualization
    visualization_msgs::MarkerArray marker_array;

    // Generate a unique ID based on timestamp to avoid overwriting previous
    // markers
    static int marker_id_counter = 0;
    int base_id = marker_id_counter;
    marker_id_counter +=
        10; // Increment by 10 to leave room for different marker types

    // Camera body marker
    visualization_msgs::Marker camera_marker;
    camera_marker.header.frame_id = frame_id_;
    camera_marker.header.stamp = ros_time;
    camera_marker.ns = "camera_bodies";
    camera_marker.id = base_id;
    camera_marker.type = visualization_msgs::Marker::CUBE;
    camera_marker.action = visualization_msgs::Marker::ADD;

    // Set the pose
    camera_marker.pose.position.x = position.x();
    camera_marker.pose.position.y = position.y();
    camera_marker.pose.position.z = position.z();
    camera_marker.pose.orientation.x = q.x();
    camera_marker.pose.orientation.y = q.y();
    camera_marker.pose.orientation.z = q.z();
    camera_marker.pose.orientation.w = q.w();

    // Set the scale (size of the camera visualization)
    camera_marker.scale.x = 0.2; // Increased width from 0.1
    camera_marker.scale.y = 0.2; // Increased height from 0.1
    camera_marker.scale.z = 0.1; // Increased thickness from 0.05

    // Set the color
    camera_marker.color.r = 0.8;
    camera_marker.color.g = 0.2;
    camera_marker.color.b = 0.2;
    camera_marker.color.a = 1.0;

    // Set the lifetime (increase to make it persist longer)
    camera_marker.lifetime = ros::Duration(0); // 0 means forever

    // Add camera marker to array
    marker_array.markers.push_back(camera_marker);

    // Create camera frustum using LINE_LIST
    visualization_msgs::Marker frustum_marker;
    frustum_marker.header.frame_id = frame_id_;
    frustum_marker.header.stamp = ros_time;
    frustum_marker.ns = "camera_frustums";
    frustum_marker.id = base_id + 1;
    frustum_marker.type = visualization_msgs::Marker::LINE_LIST;
    frustum_marker.action = visualization_msgs::Marker::ADD;
    frustum_marker.pose.orientation.w = 1.0; // Identity quaternion
    frustum_marker.scale.x = 0.05;           // Increased line width from 0.01
    frustum_marker.color.g = 0.8;
    frustum_marker.color.b = 0.8;
    frustum_marker.color.a = 1.0;
    frustum_marker.lifetime = ros::Duration(0); // 0 means forever

    // Define frustum size - make the camera model larger
    double near_plane = 0;    // Increased from 0.05
    double far_plane = 1;     // Increased from 0.3
    double fov_width = 0.5;   // Increased from 0.15
    double fov_height = 0.25; // Increased from 0.10

    // Near plane corners in camera coordinates (Z forward)
    Eigen::Vector3d near_top_right(fov_width * near_plane / far_plane,
                                   -fov_height * near_plane / far_plane,
                                   near_plane);
    Eigen::Vector3d near_top_left(-fov_width * near_plane / far_plane,
                                  -fov_height * near_plane / far_plane,
                                  near_plane);
    Eigen::Vector3d near_bottom_right(fov_width * near_plane / far_plane,
                                      fov_height * near_plane / far_plane,
                                      near_plane);
    Eigen::Vector3d near_bottom_left(-fov_width * near_plane / far_plane,
                                     fov_height * near_plane / far_plane,
                                     near_plane);

    // Far plane corners in camera coordinates (Z forward)
    Eigen::Vector3d far_top_right(fov_width, -fov_height, far_plane);
    Eigen::Vector3d far_top_left(-fov_width, -fov_height, far_plane);
    Eigen::Vector3d far_bottom_right(fov_width, fov_height, far_plane);
    Eigen::Vector3d far_bottom_left(-fov_width, fov_height, far_plane);

    // Apply camera rotation to frustum vertices
    near_top_right = rot * near_top_right + position;
    near_top_left = rot * near_top_left + position;
    near_bottom_right = rot * near_bottom_right + position;
    near_bottom_left = rot * near_bottom_left + position;

    far_top_right = rot * far_top_right + position;
    far_top_left = rot * far_top_left + position;
    far_bottom_right = rot * far_bottom_right + position;
    far_bottom_left = rot * far_bottom_left + position;

    // Helper function to add a line
    auto addLine = [&frustum_marker](const Eigen::Vector3d &start,
                                     const Eigen::Vector3d &end) {
        geometry_msgs::Point p_start, p_end;
        p_start.x = start.x();
        p_start.y = start.y();
        p_start.z = start.z();
        p_end.x = end.x();
        p_end.y = end.y();
        p_end.z = end.z();
        frustum_marker.points.push_back(p_start);
        frustum_marker.points.push_back(p_end);
    };

    // Near plane
    addLine(near_top_left, near_top_right);
    addLine(near_top_right, near_bottom_right);
    addLine(near_bottom_right, near_bottom_left);
    addLine(near_bottom_left, near_top_left);

    // Far plane
    addLine(far_top_left, far_top_right);
    addLine(far_top_right, far_bottom_right);
    addLine(far_bottom_right, far_bottom_left);
    addLine(far_bottom_left, far_top_left);

    // Connect near and far planes
    addLine(near_top_left, far_top_left);
    addLine(near_top_right, far_top_right);
    addLine(near_bottom_right, far_bottom_right);
    addLine(near_bottom_left, far_bottom_left);

    // Add coordinate axes marker
    visualization_msgs::Marker axes_marker;
    axes_marker.header.frame_id = frame_id_;
    axes_marker.header.stamp = ros_time;
    axes_marker.ns = "camera_axes";
    axes_marker.id = base_id + 2;
    axes_marker.type = visualization_msgs::Marker::LINE_LIST;
    axes_marker.action = visualization_msgs::Marker::ADD;
    axes_marker.pose.position = camera_marker.pose.position;
    axes_marker.pose.orientation = camera_marker.pose.orientation;
    axes_marker.scale.x = 0.02;              // Increased line width from 0.02
    axes_marker.lifetime = ros::Duration(0); // 0 means forever

    // Define axis lengths
    double axis_length = 0.25; // Increased from 0.15

    // Define colors for axes (X=red, Y=green, Z=blue)
    std_msgs::ColorRGBA red, green, blue;
    red.r = 1.0;
    red.a = 1.0;
    green.g = 1.0;
    green.a = 1.0;
    blue.b = 1.0;
    blue.a = 1.0;

    // Add axis lines
    geometry_msgs::Point origin, x_end, y_end, z_end;
    origin.x = origin.y = origin.z = 0.0;
    x_end.x = axis_length;
    x_end.y = x_end.z = 0.0;
    y_end.y = axis_length;
    y_end.x = y_end.z = 0.0;
    z_end.z = axis_length;
    z_end.x = z_end.y = 0.0;

    axes_marker.points.push_back(origin);
    axes_marker.points.push_back(x_end);
    axes_marker.points.push_back(origin);
    axes_marker.points.push_back(y_end);
    axes_marker.points.push_back(origin);
    axes_marker.points.push_back(z_end);

    axes_marker.colors.push_back(red);
    axes_marker.colors.push_back(red);
    axes_marker.colors.push_back(green);
    axes_marker.colors.push_back(green);
    axes_marker.colors.push_back(blue);
    axes_marker.colors.push_back(blue);

    // Add frustum and axes markers to array
    marker_array.markers.push_back(frustum_marker);
    marker_array.markers.push_back(axes_marker);

    // Publish markers
    camera_pose_pub_.publish(marker_array);

    // Update and publish camera trajectory
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = frame_id_;
    pose_stamped.header.stamp = ros_time;
    pose_stamped.pose.position.x = position.x();
    pose_stamped.pose.position.y = position.y();
    pose_stamped.pose.position.z = position.z();
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    camera_trajectory_.push_back(pose_stamped);

    // Keep trajectory at a reasonable size
    if (camera_trajectory_.size() > 1000) {
        camera_trajectory_.erase(camera_trajectory_.begin());
    }

    // Publish path for trajectory visualization
    nav_msgs::Path path_msg;
    path_msg.header.frame_id = frame_id_;
    path_msg.header.stamp = ros_time;
    path_msg.poses = camera_trajectory_;

    camera_trajectory_pub_.publish(path_msg);
}

void CalibProcessor::publishDepthImage(const cv::Mat &depth_image, double timestamp) {
    if (!ros_initialized_ || !nh_) {
        return;
    }

    // Convert OpenCV image to ROS message
    sensor_msgs::ImagePtr img_msg =
        cv_bridge::CvImage(std_msgs::Header(), "mono8", depth_image).toImageMsg();

    // Set timestamp
    img_msg->header.stamp = ros::Time(timestamp);
    img_msg->header.frame_id = "camera";

    // Publish the image
    depth_image_pub_.publish(img_msg);

    ROS_DEBUG("Published depth image");
}

void CalibProcessor::publishMatchImage(const std::string &image_path,
                                       double timestamp) {
    if (!ros_initialized_ || !nh_) {
        return;
    }

    // Load the match visualization image
    cv::Mat match_image = cv::imread(image_path);
    if (match_image.empty()) {
        ROS_WARN("Failed to load match visualization image: %s",
                 image_path.c_str());
        return;
    }

    // Convert OpenCV image to ROS message
    sensor_msgs::ImagePtr img_msg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", match_image).toImageMsg();

    // Set timestamp
    img_msg->header.stamp = ros::Time(timestamp);
    img_msg->header.frame_id = "camera";

    // Publish the image
    match_image_pub_.publish(img_msg);

    ROS_DEBUG("Published match visualization image");
}

void CalibProcessor::publishImage(const cv::Mat &image, double timestamp) {
    if (!ros_initialized_ || !nh_) {
        return;
    }

    // Convert OpenCV image to ROS message
    sensor_msgs::ImagePtr img_msg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

    // Set timestamp
    img_msg->header.stamp = ros::Time(timestamp);
    img_msg->header.frame_id = "camera";

    // Publish the image
    image_pub_.publish(img_msg);

    ROS_DEBUG("Published image");
}
} // namespace lvmapping
