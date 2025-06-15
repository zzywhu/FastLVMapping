#include "calib.h"
#include "config.h"
#include "ImageProcess/imageprocess.h"
#include <pcl/io/pcd_io.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;
namespace lvmapping {

bool compareTimestamps(const std::pair<double, Eigen::Matrix4d>& a, const std::pair<double, Eigen::Matrix4d>& b) {
    return a.first < b.first;
}

CalibProcessor::CalibProcessor() {
    // Default initialization using configuration
    Config& config = Config::getInstance();
    
    camera_matrix_ = config.cameraParams().camera_matrix.clone();
    newcamera_matrix_ = config.cameraParams().new_camera_matrix.clone();
    resizecamera_matrix_ = config.cameraParams().resize_camera_matrix.clone();
    dist_coeffs_ = config.cameraParams().distortion_coeffs.clone();
    
    T_lidar_camera_ = config.calibParams().T_lidar_camera;
    T_lidar_camera_update_ = T_lidar_camera_; // Initialize with the same value
}

CalibProcessor::~CalibProcessor() {
    // Nothing specific to do in destructor
}

bool CalibProcessor::loadCalibrationParameters(const std::string& config_file) {
    Config& config = Config::getInstance();
    if (!config.loadFromYAML(config_file)) {
        std::cerr << "Failed to load configuration from " << config_file << std::endl;
        return false;
    }
    
    // Update member variables from loaded config
    camera_matrix_ = config.cameraParams().camera_matrix.clone();
    newcamera_matrix_ = config.cameraParams().new_camera_matrix.clone();
    resizecamera_matrix_ = config.cameraParams().resize_camera_matrix.clone();
    std::cout<<config.cameraParams().resize_camera_matrix<<std::endl;
    dist_coeffs_ = config.cameraParams().distortion_coeffs.clone();
    T_lidar_camera_ = config.calibParams().T_lidar_camera;
    T_lidar_camera_update_ = T_lidar_camera_;
    
    visualization_tool_ = config.outputParams().visualization_tool;
    
    return true;
}

bool CalibProcessor::saveCalibrationParameters(const std::string& config_file) {
    Config& config = Config::getInstance();
    
    // Update config with current values
    config.mutableCameraParams().camera_matrix = camera_matrix_.clone();
    config.mutableCameraParams().new_camera_matrix = newcamera_matrix_.clone();
    config.mutableCameraParams().resize_camera_matrix = resizecamera_matrix_.clone();
    config.mutableCameraParams().distortion_coeffs = dist_coeffs_.clone();
    
    config.mutableCalibParams().T_lidar_camera = T_lidar_camera_update_;
    
    config.saveToYAML(config_file);
    return true;
}

void CalibProcessor::setVisualizationTool(const std::string& tool_path) {
    visualization_tool_ = tool_path;
    Config::getInstance().mutableOutputParams().visualization_tool = tool_path;
}

bool CalibProcessor::interpolatePose(double timestamp, 
                                    const std::vector<std::pair<double, Eigen::Matrix4d>>& trajectory,
                                    Eigen::Matrix4d& lidar_pose) {
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
bool CalibProcessor::updateExtrinsics(const std::string& match_file,
                                     const std::string& index_file, 
                                     const Eigen::Matrix4d& lidar_pose,
                                     const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    std::cout << "Using match file: " << match_file << " and index file: " << index_file 
              << " to update extrinsics" << std::endl;
    
    // Get configuration parameters
    const Config& config = Config::getInstance();
    const auto& pnp_params = config.pnpParams();
    
    // Read match file
    std::vector<std::pair<cv::Point2f, cv::Point2f>> matches;
    std::ifstream match_in(match_file);
    if (!match_in.is_open()) {
        std::cerr << "Failed to open match file: " << match_file << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(match_in, line)) {
        std::replace(line.begin(), line.end(), ',', ' '); // Replace commas with spaces
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
    index_in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    index_in.read(reinterpret_cast<char*>(&cols), sizeof(int));
    
    index_map = cv::Mat(rows, cols, CV_32S);
    index_in.read(reinterpret_cast<char*>(index_map.data), rows * cols * sizeof(int));
    index_in.close();
    
    // Calculate camera pose from LiDAR pose and extrinsics
    Eigen::Matrix4d camera_pose = lidar_pose * T_lidar_camera_.inverse();
    
    // Transform point cloud to camera coordinate system
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        const auto& point = cloud->points[i];
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
    
    for (const auto& match : matches) {
        int x1 = static_cast<int>(match.first.x);
        int y1 = static_cast<int>(match.first.y);
        
        if (x1 >= 0 && x1 < index_map.cols && y1 >= 0 && y1 < index_map.rows) {
            int point_idx = index_map.at<int>(y1, x1);
            
            if (point_idx >= 0 && point_idx < transformed_cloud->points.size()) {
                const auto& point = transformed_cloud->points[point_idx];
                obj_points.push_back(cv::Point3f(point.x, point.y, point.z));
                img_points.push_back(match.second);
            }
        }
    }
    
    if (obj_points.size() < pnp_params.min_inlier_count) {
        std::cout << "Not enough valid correspondences for PnP: " << obj_points.size() << std::endl;
        return false;
    }
    
    std::cout << "Found " << obj_points.size() << " valid 3D-2D correspondences for PnP" << std::endl;
    
    // Solve PnP
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat inliers;
    
    bool pnp_success = cv::solvePnPRansac(
        obj_points,
        img_points,
        resizecamera_matrix_,
        cv::Mat::zeros(1, 5, CV_64F),
        rvec,
        tvec,
        true,
        pnp_params.ransac_iterations,
        pnp_params.reprojection_error,
        pnp_params.confidence,
        inliers,
        cv::SOLVEPNP_ITERATIVE
    );
    if (pnp_success) {
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
        
        if (rotation_diff > pnp_params.max_rotation_diff || 
            translation_diff > pnp_params.max_translation_diff) {
            std::cout << "Warning: Transformation change too large, skipping this update!" << std::endl;
            return false;
        }
        
        // Update LiDAR to camera transformation matrix
        T_lidar_camera_update_ = T_delta * T_lidar_camera_;
        
        std::cout << "Updated extrinsics from PnP:" << std::endl;
        std::cout << T_lidar_camera_update_ << std::endl;
        
        return true;
    } else {
        std::cerr << "PnP failed" << std::endl;
        return false;
    }
}

bool CalibProcessor::colorizePointCloud(const cv::Mat& undistorted_img,
                                       const Eigen::Matrix4d& camera_pose,
                                       const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colored_cloud,
                                       std::vector<int>& point_color_count,
                                       const std::string& output_folder,
                                       double timestamp) {
    // Get configuration parameters
    const Config& config = Config::getInstance();
    const auto& pc_params = config.pointCloudParams();
    const auto& proj_params = config.projectionParams();
    
    // Create depth visualization image
    cv::Mat depth_viz_img = undistorted_img.clone();
    
    // Create depth buffer and point index buffer
    cv::Mat depth_buffer(depth_viz_img.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
    cv::Mat point_index_buffer(depth_viz_img.size(), CV_32S, cv::Scalar(-1));
    
    // First pass: Fill depth buffer
    for (size_t i = 0; i < cloud->points.size(); i++) {
        Eigen::Vector4d pt_world(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1.0);
        Eigen::Vector4d pt_camera = camera_pose.inverse() * pt_world;
        
        // Skip points behind camera or too far
        if (pt_camera[2] <= 0 || pt_camera[2] >= pc_params.max_depth) continue;
        
        // Project to image plane
        int px = static_cast<int>(proj_params.focal_length * pt_camera[0] / pt_camera[2] + proj_params.image_center_x);
        int py = static_cast<int>(proj_params.focal_length * pt_camera[1] / pt_camera[2] + proj_params.image_center_y);
        
        // Check if within valid image bounds
        if (px >= proj_params.valid_image_start_x && px < proj_params.valid_image_end_x && 
            py >= proj_params.valid_image_start_y && py < proj_params.valid_image_end_y) {
            
            float depth = static_cast<float>(pt_camera[2]);
            
            // Update neighborhood pixels
            for (int ny = std::max(proj_params.valid_image_start_y, py - pc_params.neighborhood_size); 
                 ny <= std::min(proj_params.valid_image_end_y - 1, py + pc_params.neighborhood_size); ny++) {
                for (int nx = std::max(proj_params.valid_image_start_x, px - pc_params.neighborhood_size); 
                     nx <= std::min(proj_params.valid_image_end_x - 1, px + pc_params.neighborhood_size); nx++) {
                    
                    // If current point is closer than what's in the buffer
                    if (depth < depth_buffer.at<float>(ny, nx)) {
                        depth_buffer.at<float>(ny, nx) = depth;
                        point_index_buffer.at<int>(ny, nx) = static_cast<int>(i);
                    }
                }
            }
        }
    }
    
    // Second pass: Colorize points and create visualization
    for (int y = proj_params.valid_image_start_y; y < proj_params.valid_image_end_y; y++) {
        for (int x = proj_params.valid_image_start_x; x < proj_params.valid_image_end_x; x++) {
            int point_idx = point_index_buffer.at<int>(y, x);
            if (point_idx >= 0) {
                // Get depth value
                float depth = depth_buffer.at<float>(y, x);
                
                // Create depth visualization
                float normalized_depth = (depth - pc_params.min_depth) / (pc_params.max_depth - pc_params.min_depth);
                normalized_depth = std::min(std::max(normalized_depth, 0.0f), 1.0f);
                
                int r = static_cast<int>((1.0f - normalized_depth) * 255);
                int g = static_cast<int>((normalized_depth > 0.5f ? 
                                        1.0f - normalized_depth : normalized_depth) * 255);
                int b = static_cast<int>(normalized_depth * 255);
                
                depth_viz_img.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
                
                // Colorize point cloud
                cv::Vec3b color = undistorted_img.at<cv::Vec3b>(y, x);
                colored_cloud->points[point_idx].b = color[0];
                colored_cloud->points[point_idx].g = color[1];
                colored_cloud->points[point_idx].r = color[2];
                point_color_count[point_idx]++;
            }
        }
    }
    
    // Save depth visualization image
    std::string depth_viz_dir = output_folder + "/depth_viz";
    if (!fs::exists(depth_viz_dir)) {
        fs::create_directories(depth_viz_dir);
    }
    
    std::string depth_viz_path = depth_viz_dir + "/" + std::to_string(timestamp) + "_depth.png";
    cv::imwrite(depth_viz_path, depth_viz_img);
    
    return true;
}

bool CalibProcessor::saveVisualization(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colored_cloud,
                                      const std::vector<int>& point_color_count,
                                      const std::string& output_folder,
                                      const std::string& progress_info) {
    // Create directory for colormap if it doesn't exist
    std::string colormap_dir = output_folder + "/colormap";
    if (!fs::exists(colormap_dir)) {
        fs::create_directories(colormap_dir);
    }
    
    // Display progress info
    std::cout << progress_info << std::endl;
    
    // Display point coloring statistics
    int colored_points = 0;
    for (size_t i = 0; i < point_color_count.size(); ++i) {
        if (point_color_count[i] > 0) {
            colored_points++;
        }
    }
    
    double percentage = (100.0 * colored_points / colored_cloud->points.size());
    std::cout << "Points colored so far: " << colored_points << "/" << colored_cloud->points.size()
              << " (" << percentage << "%)" << std::endl;
    
    // Optionally, launch PCL viewer
    static bool viewer_launched = false;
    if (!viewer_launched) {
        if (!visualization_tool_.empty()) {
            std::cout << "To watch the coloring process in real-time, you can run:" << std::endl;
            std::cout << visualization_tool_ << " " << colormap_dir << "/colored_pointcloud_live.pcd" << std::endl;
        } else {
            std::cout << "No point cloud viewer available. Install pcl-tools, CloudCompare, or MeshLab for visualization." << std::endl;
        }
        viewer_launched = true;
    }
    
    // Create a visualization cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr viz_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*colored_cloud));
    for (size_t i = 0; i < viz_cloud->points.size(); ++i) {
        if (point_color_count[i] > 0) {
            viz_cloud->points[i].r = point_color_count[i];
            viz_cloud->points[i].g = point_color_count[i];
            viz_cloud->points[i].b = point_color_count[i];
        }
    }
    
    // Save the visualization file
    pcl::io::savePCDFileBinary(colormap_dir + "/colored_pointcloud_live.pcd", *viz_cloud);
    
    // Create and save a cloud with only colored points
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_only_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (size_t i = 0; i < colored_cloud->points.size(); ++i) {
        if (point_color_count[i] > 0) {
            colored_only_cloud->push_back(colored_cloud->points[i]);
        }
    }
    
    pcl::io::savePCDFileBinary(colormap_dir + "/colored_only_points_live.pcd", *colored_only_cloud);
    
    return true;
}

bool CalibProcessor::processImageFrame(double timestamp, 
                                      const std::string& img_path, 
                                      const Eigen::Matrix4d& lidar_pose,
                                      const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colored_cloud,
                                      std::vector<int>& point_color_count,
                                      const std::string& output_folder) {
    // Load image
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "Failed to open image: " << img_path << std::endl;
        return false;
    }
    
    // Undistort the fisheye image
    cv::Mat undistorted_img;
    cv::fisheye::undistortImage(img, undistorted_img, camera_matrix_, dist_coeffs_, newcamera_matrix_);
    
    // Calculate camera pose
    Eigen::Matrix4d camera_pose = lidar_pose * T_lidar_camera_update_.inverse();
    
    // Colorize point cloud with depth-aware method
    return colorizePointCloud(undistorted_img, camera_pose, cloud, colored_cloud, 
                             point_color_count, output_folder, timestamp);
}

bool CalibProcessor::processImagesAndPointCloud(const std::string& image_folder, 
                                              const std::string& trajectory_file,
                                              const std::string& pcd_file,
                                              const std::string& output_folder) {
    // Get configuration parameters
    const Config& config = Config::getInstance();
    const int viz_frequency = config.outputParams().viz_frequency;
    const int img_sampling_step = config.processingParams().img_sampling_step;
    
    // Create output directory if it doesn't exist
    if (!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }
    
    // Load calibration parameters from default configuration file
    std::string config_file = "/home/zzy/SensorCalibration/FastLVMapping/config/default_config.yaml";
    if (camera_matrix_.empty()) {
        if (!loadCalibrationParameters(config_file)) {
            std::cerr << "Failed to load calibration parameters from default config: " << config_file << std::endl;
            std::cerr << "Falling back to camera_intrinsics.yaml..." << std::endl;
        }
    }
    
    // Load trajectory data
    std::vector<std::pair<double, Eigen::Matrix4d>> trajectory;
    std::ifstream traj_file(trajectory_file);
    if (!traj_file.is_open()) {
        std::cerr << "Failed to open trajectory file: " << trajectory_file << std::endl;
        return false;
    }
    
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
    
    // Sort trajectory by timestamp
    std::sort(trajectory.begin(), trajectory.end(), compareTimestamps);
    
    // Load point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *cloud) == -1) {
        std::cerr << "Failed to load point cloud: " << pcd_file << std::endl;
        return false;
    }
    
    // Create the colored point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored_cloud->points.resize(cloud->points.size());
    colored_cloud->width = cloud->width;
    colored_cloud->height = cloud->height;
    
    // Initialize the colored cloud with coordinates
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        colored_cloud->points[i].x = cloud->points[i].x;
        colored_cloud->points[i].y = cloud->points[i].y;
        colored_cloud->points[i].z = cloud->points[i].z;
        colored_cloud->points[i].r = 0;
        colored_cloud->points[i].g = 0;
        colored_cloud->points[i].b = 0;
    }
    
    // Create a counter for each point
    std::vector<int> point_color_count(cloud->points.size(), 0);
    
    // Get all image files from directory
    std::vector<std::pair<double, std::string>> image_files;
    for (const auto& entry : fs::directory_iterator(image_folder)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            // Assuming filename is the timestamp
            double timestamp = std::stod(filename.substr(0, filename.find_last_of('.')));
            image_files.push_back({timestamp, entry.path().string()});
        }
    }
    
    // Sort images by timestamp
    std::sort(image_files.begin(), image_files.end(), 
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // Sample images to improve processing speed
    std::vector<std::pair<double, std::string>> selected_images;
    for (size_t i = 0; i < image_files.size(); i += img_sampling_step) {
        selected_images.push_back(image_files[i]);
    }
    
    std::cout << "Selected " << selected_images.size() << " images out of " << image_files.size() 
              << " for processing (sampling every " << img_sampling_step << " images)" << std::endl;
    
    // Get match and index file folders
    std::string match_folder = output_folder + "/match/coordinates";
    std::string index_folder = output_folder + "/index_maps";
    
    if (!fs::exists(match_folder) || !fs::exists(index_folder)) {
        std::cerr << "Match or index folders not found. Please run preprocess first." << std::endl;
        return false;
    }
    
    // Collect all match and index files
    std::map<double, std::string> match_files;
    std::map<double, std::string> index_files;
    
    // Read match files
    for (const auto& entry : fs::directory_iterator(match_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::string filename = entry.path().filename().string();
            double timestamp = std::stod(filename.substr(0, filename.find_last_of('.')));
            match_files[timestamp] = entry.path().string();
        }
    }
    
    // Read index files
    for (const auto& entry : fs::directory_iterator(index_folder)) {
        if (entry.is_regular_file() && entry.path().filename().string().find("_index.bin") != std::string::npos) {
            std::string filename = entry.path().filename().string();
            double timestamp = std::stod(filename.substr(0, filename.find_last_of('_')));
            index_files[timestamp] = entry.path().string();
        }
    }
    
    std::cout << "Found " << match_files.size() << " match files and " 
              << index_files.size() << " index files" << std::endl;
    
    // Flag to track if extrinsics were updated
    bool extrinsics_updated = false;
    
    // Process each selected image
    int img_count = 0;
    for (const auto& img_data : selected_images) {
        double timestamp = img_data.first;
        std::string img_path = img_data.second;
        
        // Interpolate LiDAR pose at this timestamp
        Eigen::Matrix4d lidar_pose;
        if (!interpolatePose(timestamp, trajectory, lidar_pose)) {
            continue;
        }
        
        // Check if there are matching files for calibration update
        if (match_files.find(timestamp) != match_files.end() && 
            index_files.find(timestamp) != index_files.end()) {
            
            if (!updateExtrinsics(match_files[timestamp], index_files[timestamp], lidar_pose, cloud)) {
                // After updating extrinsics, save the new calibration
                //saveCalibrationParameters(config_file);
                continue;
            }
        }
        
        // Process image frame for colorization
        if (!processImageFrame(timestamp, img_path, lidar_pose, cloud, colored_cloud, point_color_count, output_folder)) {
            std::cerr << "Failed to process image frame: " << img_path << std::endl;
            continue;
        }
        
        // Show progress and save visualization periodically
        if (img_count % viz_frequency == 0) {
            std::string progress_info = "Processed " + std::to_string(img_count + 1) + "/" + 
                                      std::to_string(selected_images.size()) + " images. Progress: " + 
                                      std::to_string(100.0 * (img_count + 1) / selected_images.size()) + "%";
            saveVisualization(colored_cloud, point_color_count, output_folder, progress_info);
        }
        
        img_count++;
    }
    
    // Final save and report
    std::string progress_info = "Processing complete. Processed " + std::to_string(img_count) + 
                              " images out of " + std::to_string(selected_images.size());
    saveVisualization(colored_cloud, point_color_count, output_folder, progress_info);
    
    
    return true;
}
bool CalibProcessor::preprocess(const std::string& image_folder, 
                                              const std::string& trajectory_file,
                                              const std::string& pcd_file,
                                              const std::string& output_folder) {
    // Create output directory if not exists
    if (!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }
    
    // Use default configuration file
    std::string config_file = "/home/zzy/SensorCalibration/FastLVMapping/config/default_config.yaml";
    
    // Load calibration parameters
    if (!loadCalibrationParameters(config_file)) {
        std::cerr << "Failed to load calibration parameters from default config: " << config_file << std::endl;
        std::cerr << "Falling back to camera_intrinsics.yaml..." << std::endl;
        
        // Fall back to the original config file
        config_file = "/home/zzy/SensorCalibration/FastLVMapping/config/camera_intrinsics.yaml";
        if (!loadCalibrationParameters(config_file)) {
            std::cerr << "Failed to load calibration parameters from fallback config" << std::endl;
            return false;
        }
    }
    
    // Load trajectory data
    std::vector<std::pair<double, Eigen::Matrix4d>> trajectory;
    std::ifstream traj_file(trajectory_file);
    if (!traj_file.is_open()) {
        std::cerr << "Failed to open trajectory file: " << trajectory_file << std::endl;
        return false;
    }
    
    // Parse trajectory file (time, x, y, z, qx, qy, qz, qw)
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
    // Sort trajectory by timestamp
    std::sort(trajectory.begin(), trajectory.end(), compareTimestamps);
    
    // Load point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *cloud) == -1) {
        std::cerr << "Failed to load point cloud: " << pcd_file << std::endl;
        return false;
    }
    // Create a counter for each point to average colors from multiple images
    
    // Get all image files from directory
    std::vector<std::pair<double, std::string>> image_files;
    for (const auto& entry : fs::directory_iterator(image_folder)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            // Assuming filename is the timestamp
            double timestamp = std::stod(filename.substr(0, filename.find_last_of('.')));
            image_files.push_back({timestamp, entry.path().string()});
        }
    }
    
    // Sort images by timestamp
    std::sort(image_files.begin(), image_files.end(), 
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // 隔一张图片处理，提高处理速度
    std::vector<std::pair<double, std::string>> selected_images;
    for (size_t i = 0; i < image_files.size(); i += 2) {  // 步长为2，表示隔一张取一张
        selected_images.push_back(image_files[i]);
    }
    
    std::cout << "Selected " << selected_images.size() << " images out of " << image_files.size() 
              << " for processing (processing every other image)" << std::endl;
    
    // Process each selected image
    int processed_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (const auto& img_data : selected_images) {
        double timestamp = img_data.first;
        std::string img_path = img_data.second;
        
        // Load image
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << img_path << std::endl;
            continue;
        }
        
       
        
        // Find camera pose at this timestamp by interpolation
        Eigen::Matrix4d lidar_pose = Eigen::Matrix4d::Identity();
        if (timestamp <= trajectory.front().first) {
            continue;
        } else if (timestamp >= trajectory.back().first) {
            continue;
        } else {
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
            
            double alpha = (timestamp - t1) / (t2 - t1);
            
            // Position interpolation
            Eigen::Vector3d pos1 = pose1.block<3, 1>(0, 3);
            Eigen::Vector3d pos2 = pose2.block<3, 1>(0, 3);
            Eigen::Vector3d pos = pos1 + alpha * (pos2 - pos1);
            
            // Rotation interpolation
            Eigen::Quaterniond q1(pose1.block<3, 3>(0, 0));
            Eigen::Quaterniond q2(pose2.block<3, 3>(0, 0));
            Eigen::Quaterniond q = q1.slerp(alpha, q2);
            
            lidar_pose.block<3, 3>(0, 0) = q.toRotationMatrix();
            lidar_pose.block<3, 1>(0, 3) = pos;
        }
        // Calculate camera pose from LiDAR pose
        Eigen::Matrix4d camera_pose = lidar_pose * T_lidar_camera_.inverse();
        
        // Transform point cloud to camera coordinate system for projection
        // We'll project original point cloud points to image to update the colored cloud
        ImageProcess imgProc;
        
        
        // Undistort the fisheye image
        cv::Mat undistorted_img;
        cv::fisheye::undistortImage(img, undistorted_img, camera_matrix_, dist_coeffs_,newcamera_matrix_);
        
        cv::Mat resized_img;
     
        cv::resize(undistorted_img, resized_img, cv::Size(800, 800), 0, 0, cv::INTER_LINEAR);
        std::string undist_dir = output_folder + "/undist_img";
         if (!fs::exists(undist_dir)) {
            fs::create_directories(undist_dir);
        }
        std::string undist_img_path = undist_dir + "/" + std::to_string(timestamp) + ".png";
        cv::imwrite(undist_img_path, resized_img);  // 保存原始去畸变图像



        // Generate and save the projected image for visualization (same as before)
        pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto& point : cloud->points) {
            Eigen::Vector4d pt_world(point.x, point.y, point.z, 1.0);
            Eigen::Vector4d pt_camera = camera_pose.inverse() * pt_world;
            
            pcl::PointXYZI transformed_point;
            transformed_point.x = pt_camera[0];
            transformed_point.y = pt_camera[1];
            transformed_point.z = pt_camera[2];
            transformed_point.intensity = point.intensity;
            
            transformed_cloud->push_back(transformed_point);
        }
        

        // Project the point cloud to the image plane for visualization and get index mapping
        auto [projected_image, index_map] = imgProc.projectPinholeWithIndices(transformed_cloud, true);
        
        // Save the projected image in the rimg folder with timestamp
        std::string rimg_dir = output_folder + "/rimg";
        if (!fs::exists(rimg_dir)) {
            fs::create_directories(rimg_dir);
        }
        
        std::string projected_img_path = rimg_dir + "/" + std::to_string(timestamp) + ".png";
        cv::imwrite(projected_img_path, projected_image);
        
        // Save the index map as image
        std::string index_dir = output_folder + "/index_maps";
        if (!fs::exists(index_dir)) {
            fs::create_directories(index_dir);
        }
        
        // Convert index map to a visualization-friendly format
        // Using multiple channels to store the full range of indices
        cv::Mat index_image(index_map.size(), CV_16UC3);
        
        // Find max index for normalization (for viewing purposes)
        int max_index = 0;
        for (int y = 0; y < index_map.rows; y++) {
            for (int x = 0; x < index_map.cols; x++) {
                int idx = index_map.at<int>(y, x);
                if (idx > max_index) max_index = idx;
            }
        }
        
        // Convert each index to color components that can encode the full index value
        for (int y = 0; y < index_map.rows; y++) {
            for (int x = 0; x < index_map.cols; x++) {
                int idx = index_map.at<int>(y, x);
                if (idx >= 0) {  // Valid point index
                    // Encode the index as 3 16-bit channels
                    // This allows us to encode indices up to 2^48 - 1
                    index_image.at<cv::Vec3w>(y, x)[0] = idx & 0xFFFF;                 // Lower 16 bits
                    index_image.at<cv::Vec3w>(y, x)[1] = (idx >> 16) & 0xFFFF;        // Middle 16 bits
                    index_image.at<cv::Vec3w>(y, x)[2] = (idx >> 32) & 0xFFFF;        // Upper 16 bits
                } else {
                    // No point maps to this pixel
                    index_image.at<cv::Vec3w>(y, x) = cv::Vec3w(0, 0, 0);
                }
            }
        }
        
        // Save the raw index image
        std::string index_map_path = index_dir + "/" + std::to_string(timestamp) + "_index.png";
        cv::imwrite(index_map_path, index_image);
        
        // Also create a visualization image for easier debugging
        cv::Mat vis_index_map(index_map.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        for (int y = 0; y < index_map.rows; y++) {
            for (int x = 0; x < index_map.cols; x++) {
                int idx = index_map.at<int>(y, x);
                if (idx >= 0) {  // Valid point index
                    // Create a color based on the index (for visualization only)
                    float normalized_idx = static_cast<float>(idx) / max_index;
                    int hue = static_cast<int>(normalized_idx * 180);  // Hue ranges from 0 to 180 in OpenCV
                    
                    // Convert HSV to RGB for visualization (fully saturated, full value)
                    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
                    cv::Mat rgb;
                    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
                    
                    vis_index_map.at<cv::Vec3b>(y, x) = rgb.at<cv::Vec3b>(0, 0);
                }
            }
        }
        
        // Save the visualization image
        std::string vis_index_path = index_dir + "/" + std::to_string(timestamp) + "_index_vis.png";
        cv::imwrite(vis_index_path, vis_index_map);
        
        // Also still save the binary version for efficient loading
        std::string binary_index_path = index_dir + "/" + std::to_string(timestamp) + "_index.bin";
        std::ofstream index_file(binary_index_path, std::ios::binary);
        if (index_file.is_open()) {
            // Write matrix dimensions
            int rows = index_map.rows;  // 获取矩阵的行数
            int cols = index_map.cols;  // 获取矩阵的列数
            
            // 将矩阵的行数写入文件
            // reinterpret_cast将整数指针转换为char*类型，以便按字节写入
            // sizeof(int)确保写入正确的字节数（通常为4字节）
            index_file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
            
            // 将矩阵的列数写入文件
            index_file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
            
            // 写入矩阵的实际数据
            // index_map.data指向矩阵的数据区域的开始
            // rows * cols * sizeof(int)计算了整个矩阵的字节大小
            index_file.write(reinterpret_cast<const char*>(index_map.data), rows * cols * sizeof(int));
            index_file.close();  // 关闭文件
        } else {
            std::cerr << "Failed to save index map to: " << index_map_path << std::endl;
        }
        
        // Show progress feedback
        processed_count++;
        if (processed_count % 10 == 0 || processed_count == selected_images.size()) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            double progress = 100.0 * processed_count / selected_images.size();
            
            std::cout << "Preprocessing progress: " << processed_count << "/" << selected_images.size()
                      << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::endl;
            
            if (elapsed > 0 && processed_count > 10) {
                double images_per_second = static_cast<double>(processed_count) / elapsed;
                double remaining_seconds = (selected_images.size() - processed_count) / images_per_second;
                
                int minutes = static_cast<int>(remaining_seconds) / 60;
                int seconds = static_cast<int>(remaining_seconds) % 60;
                
                std::cout << "Processing speed: " << std::fixed << std::setprecision(2) 
                          << images_per_second << " images/sec" << std::endl;
                std::cout << "Estimated time remaining: " << minutes << " minutes " 
                          << seconds << " seconds" << std::endl;
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    int minutes = static_cast<int>(total_time) / 60;
    int seconds = static_cast<int>(total_time) % 60;
    
    std::cout << "Preprocessing complete! Processed " << processed_count << " images in "
              << minutes << " minutes and " << seconds << " seconds." << std::endl;
    
    return true;
}

bool CalibProcessor::initialize(const std::string& config_file) {
    // Load configuration
    Config& config = Config::getInstance();
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
    
    return true;
}

bool CalibProcessor::run() {
    std::cout << "Starting processing with parameters:" << std::endl;
    std::cout << "Image folder: " << image_folder_ << std::endl;
    std::cout << "Trajectory file: " << trajectory_file_ << std::endl;
    std::cout << "Point cloud file: " << pcd_file_ << std::endl;
    std::cout << "Output folder: " << output_folder_ << std::endl;
    
    // Run the main processing
    return processImagesAndPointCloud(image_folder_, trajectory_file_, pcd_file_, output_folder_);
}

bool CalibProcessor::runPreprocessing() {
    std::cout << "Starting preprocessing with parameters:" << std::endl;
    std::cout << "Image folder: " << image_folder_ << std::endl;
    std::cout << "Trajectory file: " << trajectory_file_ << std::endl;
    std::cout << "Point cloud file: " << pcd_file_ << std::endl;
    std::cout << "Output folder: " << output_folder_ << std::endl;
    
    // Run preprocessing
    return preprocess(image_folder_, trajectory_file_, pcd_file_, output_folder_);
}

} // namespace lvmapping
