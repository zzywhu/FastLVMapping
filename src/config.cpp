#include "config.h"
#include <iostream>

namespace lvmapping {

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

Config::Config() {
    setDefaultValues();
}

void Config::setDefaultValues() {
    // Set default camera parameters
    camera_params_.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_params_.new_camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_params_.resize_camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_params_.distortion_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    camera_params_.camera_model = "fisheye";  // Default to fisheye
    
    // Set default calibration parameters
    calib_params_.T_lidar_camera = Eigen::Matrix4d::Identity();
    
    // Set default point cloud parameters
    pc_params_.min_depth = 0.1f;
    pc_params_.max_depth = 50.0f;
    pc_params_.neighborhood_size = 2;
    
    // Set default projection parameters
    proj_params_.image_width = 3000;
    proj_params_.image_height = 3000;
    proj_params_.focal_length = 1500;
    proj_params_.image_center_x = 1500;
    proj_params_.image_center_y = 1500;
    proj_params_.valid_image_start_x = 500;
    proj_params_.valid_image_start_y = 500;
    proj_params_.valid_image_end_x = 2500;
    proj_params_.valid_image_end_y = 2500;
    
    // Set default PnP parameters
    pnp_params_.max_rotation_diff = 0.5;
    pnp_params_.max_translation_diff = 0.5;
    pnp_params_.reprojection_error = 8.0;
    pnp_params_.min_inlier_count = 10;
    pnp_params_.ransac_iterations = 100;
    pnp_params_.confidence = 0.99;
    
    // Set default output parameters
    output_params_.visualization_tool = "";
    output_params_.viz_frequency = 10;
    
    // Set default processing parameters
    processing_params_.img_sampling_step = 2;
    processing_params_.pcd_path = "";
    processing_params_.img_path = "";
    processing_params_.traj_path = "";
    processing_params_.output_path = "";
}

bool Config::loadFromYAML(const std::string& config_file) {
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << config_file << std::endl;
        return false;
    }

    // Read camera model
    if (fs["camera_model"].isString())
        camera_params_.camera_model = fs["camera_model"].string();
    
    // Read camera parameters
    fs["camera_matrix"] >> camera_params_.camera_matrix;
    fs["new_camera_matrix"] >> camera_params_.new_camera_matrix;
    fs["resize_camera_matrix"] >> camera_params_.resize_camera_matrix;
    fs["distortion_coefficients"] >> camera_params_.distortion_coeffs;
    
    // Read extrinsics if available
    cv::Mat T_lidar_cam;
    if (fs["extrinsics"].isMap()) {
        fs["extrinsics"] >> T_lidar_cam;
        
        // Convert to Eigen matrix
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                calib_params_.T_lidar_camera(i, j) = T_lidar_cam.at<double>(i, j);
            }
        }
    }
    
    // Read paths
    if (fs["pcd_path"].isString())
        processing_params_.pcd_path = fs["pcd_path"].string();
    if (fs["img_path"].isString())
        processing_params_.img_path = fs["img_path"].string();
    if (fs["traj_path"].isString())
        processing_params_.traj_path = fs["traj_path"].string();
    if (fs["output_path"].isString())
        processing_params_.output_path = fs["output_path"].string();
    
    // Read other parameters from nested sections
    if (fs["point_cloud_params"].isMap()) {
        fs["point_cloud_params"]["min_depth"] >> pc_params_.min_depth;
        fs["point_cloud_params"]["max_depth"] >> pc_params_.max_depth;
        fs["point_cloud_params"]["neighborhood_size"] >> pc_params_.neighborhood_size;
    }
    
    // Read projection parameters
    if (fs["projection_params"].isMap()) {
        fs["projection_params"]["image_width"] >> proj_params_.image_width;
        fs["projection_params"]["image_height"] >> proj_params_.image_height;
        fs["projection_params"]["focal_length"] >> proj_params_.focal_length;
        fs["projection_params"]["image_center_x"] >> proj_params_.image_center_x;
        fs["projection_params"]["image_center_y"] >> proj_params_.image_center_y;
        fs["projection_params"]["valid_image_start_x"] >> proj_params_.valid_image_start_x;
        fs["projection_params"]["valid_image_start_y"] >> proj_params_.valid_image_start_y;
        fs["projection_params"]["valid_image_end_x"] >> proj_params_.valid_image_end_x;
        fs["projection_params"]["valid_image_end_y"] >> proj_params_.valid_image_end_y;
    }
    
    // Read PnP parameters
    if (fs["pnp_params"].isMap()) {
        fs["pnp_params"]["max_rotation_diff"] >> pnp_params_.max_rotation_diff;
        fs["pnp_params"]["max_translation_diff"] >> pnp_params_.max_translation_diff;
        fs["pnp_params"]["reprojection_error"] >> pnp_params_.reprojection_error;
        fs["pnp_params"]["min_inlier_count"] >> pnp_params_.min_inlier_count;
        fs["pnp_params"]["ransac_iterations"] >> pnp_params_.ransac_iterations;
        fs["pnp_params"]["confidence"] >> pnp_params_.confidence;
    }
    
    // Read output parameters
    if (fs["output_params"].isMap()) {
        fs["output_params"]["visualization_tool"] >> output_params_.visualization_tool;
        fs["output_params"]["viz_frequency"] >> output_params_.viz_frequency;
    }
    
    // Read processing parameters
    if (fs["processing_params"].isMap()) {
        fs["processing_params"]["img_sampling_step"] >> processing_params_.img_sampling_step;
    }
    
    fs.release();
    return true;
}

void Config::saveToYAML(const std::string& config_file) const {
    cv::FileStorage fs(config_file, cv::FileStorage::WRITE);
    
    // Save camera model
    fs << "camera_model" << camera_params_.camera_model;
    
    // Save camera parameters
    fs << "camera_matrix" << camera_params_.camera_matrix;
    fs << "new_camera_matrix" << camera_params_.new_camera_matrix;
    fs << "resize_camera_matrix" << camera_params_.resize_camera_matrix;
    fs << "distortion_coefficients" << camera_params_.distortion_coeffs;
    
    // Save extrinsics
    cv::Mat T_lidar_cam(4, 4, CV_64F);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            T_lidar_cam.at<double>(i, j) = calib_params_.T_lidar_camera(i, j);
        }
    }
    fs << "extrinsics" << T_lidar_cam;
    
    // Save paths
    fs << "pcd_path" << processing_params_.pcd_path;
    fs << "img_path" << processing_params_.img_path;
    fs << "traj_path" << processing_params_.traj_path;
    fs << "output_path" << processing_params_.output_path;
    
    // Save point cloud parameters
    fs << "point_cloud_params" << "{";
    fs << "min_depth" << pc_params_.min_depth;
    fs << "max_depth" << pc_params_.max_depth;
    fs << "neighborhood_size" << pc_params_.neighborhood_size;
    fs << "}";
    
    // Save projection parameters
    fs << "projection_params" << "{";
    fs << "image_width" << proj_params_.image_width;
    fs << "image_height" << proj_params_.image_height;
    fs << "focal_length" << proj_params_.focal_length;
    fs << "image_center_x" << proj_params_.image_center_x;
    fs << "image_center_y" << proj_params_.image_center_y;
    fs << "valid_image_start_x" << proj_params_.valid_image_start_x;
    fs << "valid_image_start_y" << proj_params_.valid_image_start_y;
    fs << "valid_image_end_x" << proj_params_.valid_image_end_x;
    fs << "valid_image_end_y" << proj_params_.valid_image_end_y;
    fs << "}";
    
    // Save PnP parameters
    fs << "pnp_params" << "{";
    fs << "max_rotation_diff" << pnp_params_.max_rotation_diff;
    fs << "max_translation_diff" << pnp_params_.max_translation_diff;
    fs << "reprojection_error" << pnp_params_.reprojection_error;
    fs << "min_inlier_count" << pnp_params_.min_inlier_count;
    fs << "ransac_iterations" << pnp_params_.ransac_iterations;
    fs << "confidence" << pnp_params_.confidence;
    fs << "}";
    
    // Save output parameters
    fs << "output_params" << "{";
    fs << "visualization_tool" << output_params_.visualization_tool;
    fs << "viz_frequency" << output_params_.viz_frequency;
    fs << "}";
    
    // Save processing parameters
    fs << "processing_params" << "{";
    fs << "img_sampling_step" << processing_params_.img_sampling_step;
    fs << "}";
    
    fs.release();
}

} // namespace lvmapping
