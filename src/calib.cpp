#include "calib.h"
#include "ImageProcess/imageprocess.h"
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <Eigen/Dense>
#include <iostream>

namespace fs = std::filesystem;

bool compareTimestamps(const std::pair<double, Eigen::Matrix4d>& a, const std::pair<double, Eigen::Matrix4d>& b) {
    return a.first < b.first;
}

CalibProcessor::CalibProcessor() {
    // Default initialization
    camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
    newcamera_matrix_=cv::Mat::eye(3, 3, CV_64F);
    dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64F);
    T_lidar_camera_ = Eigen::Matrix4d::Identity();
}

CalibProcessor::~CalibProcessor() {
}

bool CalibProcessor::loadCalibrationParameters(const std::string& config_file) {
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << config_file << std::endl;
        return false;
    }

    // Read camera intrinsics
    fs["camera_matrix"] >> camera_matrix_;
    fs["new_camera_matrix"] >> newcamera_matrix_;
    fs["distortion_coefficients"] >> dist_coeffs_;
    
    // Read extrinsics (LiDAR to camera transformation)
    cv::Mat T_lidar_cam;
    fs["extrinsics"] >> T_lidar_cam;
    fs.release();

    // Convert to Eigen matrix
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            T_lidar_camera_(i, j) = T_lidar_cam.at<double>(i, j);
        }
    }
    
    return true;
}

bool CalibProcessor::processImagesAndPointCloud(const std::string& image_folder, 
                                              const std::string& trajectory_file,
                                              const std::string& pcd_file,
                                              const std::string& output_folder) {
    // Create output directory if not exists
    if (!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }
    
    // Use a default config file path if config_path_ is not set
    std::string config_file = "/home/zzy/SensorCalibration/FastLVMapping/config/camera_intrinsics.yaml";
    
    // Load calibration parameters
    if (!loadCalibrationParameters(config_file)) {
        std::cerr << "Failed to load calibration parameters from " << config_file << std::endl;
        return false;
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
    
    // Create the colored point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored_cloud->points.resize(cloud->points.size());
    colored_cloud->width = cloud->width;
    colored_cloud->height = cloud->height;
    
    // Initialize the colored cloud with coordinates from original cloud (RGB will be set to black initially)
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        colored_cloud->points[i].x = cloud->points[i].x;
        colored_cloud->points[i].y = cloud->points[i].y;
        colored_cloud->points[i].z = cloud->points[i].z;
        colored_cloud->points[i].r = 0;
        colored_cloud->points[i].g = 0;
        colored_cloud->points[i].b = 0;
    }
    
    // Create a counter for each point to average colors from multiple images
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
    
    // Process each image
    for (const auto& img_data : image_files) {
        double timestamp = img_data.first;
        std::string img_path = img_data.second;
        
        // Load image
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << img_path << std::endl;
            continue;
        }
        
        // Undistort the fisheye image
        cv::Mat undistorted_img;
        cv::fisheye::undistortImage(img, undistorted_img, camera_matrix_, dist_coeffs_,newcamera_matrix_);
        
        cv::Mat resized_img;
     
        cv::resize(undistorted_img, resized_img, cv::Size(800, 800), 0, 0, cv::INTER_LINEAR);
        std::string undist_dir = output_folder + "/undist_img";
        std::string undist_img_path = undist_dir + "/" + std::to_string(timestamp) + ".png";
        cv::imwrite(undist_img_path, resized_img);  // 保存原始去畸变图像
        
        // Find camera pose at this timestamp by interpolation
        Eigen::Matrix4d lidar_pose;
        if (timestamp <= trajectory.front().first) {
            lidar_pose = trajectory.front().second;
        } else if (timestamp >= trajectory.back().first) {
            lidar_pose = trajectory.back().second;
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
        }
        
        // Calculate camera pose from LiDAR pose and extrinsics
        Eigen::Matrix4d camera_pose = lidar_pose * T_lidar_camera_.inverse();
        
        // Transform point cloud to camera coordinate system for projection
        // We'll project original point cloud points to image to update the colored cloud
        ImageProcess imgProc;
        
        // Project each point onto the undistorted image and update RGB values
        for (int i = 0; i < cloud->points.size(); i++) {
            // Transform point from world to camera frame
            Eigen::Vector4d pt_world(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1.0);
            Eigen::Vector4d pt_camera = camera_pose.inverse() * pt_world;
            
            // Skip points behind camera (negative Z)
            if (pt_camera[2] <= 0) continue;
            
            // Project 3D point to image plane
            cv::Point2d pixel;
            pixel.x = 1500 * pt_camera[0] / pt_camera[2] + 1500;
            pixel.y = 1500 * pt_camera[1] / pt_camera[2] + 1500;
            // Check if point is within image boundaries
            if (pixel.x >= 0 && pixel.x < undistorted_img.cols && 
                pixel.y >= 0 && pixel.y < undistorted_img.rows) {
                // Get RGB values from image
                cv::Vec3b color = undistorted_img.at<cv::Vec3b>(cv::Point(int(pixel.x), int(pixel.y)));
                
                // Accumulate colors for averaging (to handle multiple images seeing the same point)
                colored_cloud->points[i].b = color[0];
                colored_cloud->points[i].g = color[1];
                colored_cloud->points[i].r = color[2];
                point_color_count[i]++;
            }
        }
        
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
        
        // Project the point cloud to the image plane for visualization
        cv::Mat projected_image = imgProc.projectPinhole(transformed_cloud, true);
        
        // Save the projected image in the rimg folder with timestamp
        std::string rimg_dir = output_folder + "/rimg";
        // Create filename using the timestamp
        std::string projected_img_path = rimg_dir + "/" + std::to_string(timestamp) + ".png";
        cv::imwrite(projected_img_path, projected_image);
        
        // Add real-time visualization of the colored point cloud
        // Update visualization every 5 images or at user-defined frequency
        static int img_count = 0;
        static int viz_frequency = 10; // Adjust this value to control visualization frequency
        
        if (img_count % viz_frequency == 0) {
            // Create directory for colormap if it doesn't exist
            std::string colormap_dir = output_folder + "/colormap";
            
            // Instead of saving intermediate files, just display progress
            std::cout << "Processed " << img_count + 1 << "/" << image_files.size() 
                      << " images. Progress: " << (100.0 * (img_count + 1) / image_files.size()) << "%" << std::endl;
            
            // Display current point coloring statistics
            int colored_points = 0;
            for (size_t i = 0; i < point_color_count.size(); ++i) {
                if (point_color_count[i] > 0) {
                    colored_points++;
                }
            }
            
            double percentage = (100.0 * colored_points / cloud->points.size());
            std::cout << "Points colored so far: " << colored_points << "/" << cloud->points.size()
                      << " (" << percentage << "%)" << std::endl;
            
            // Optionally, you can launch a PCL viewer to visualize the current state of the point cloud
            // This launches only once at the beginning instead of every N images
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
            
            // Create a temporary copy with averaged colors for the live view
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr viz_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*colored_cloud));
            for (size_t i = 0; i < viz_cloud->points.size(); ++i) {
                if (point_color_count[i] > 0) {
                    // viz_cloud->points[i].r /= point_color_count[i];
                    // viz_cloud->points[i].g /= point_color_count[i];
                    // viz_cloud->points[i].b /= point_color_count[i];
                }
            }
            
            // Update the live view file (only one file that gets continuously updated)
            pcl::io::savePCDFileBinary(colormap_dir + "/colored_pointcloud_live.pcd", *viz_cloud);
        }
        
        img_count++;
    }
    
    return true;
}
