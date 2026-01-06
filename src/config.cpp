#include "config.h"
#include <iostream>

namespace lvmapping {

Config &Config::getInstance() {
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
    camera_params_.camera_model = "fisheye"; // Default to fisheye

    // Set default calibration parameters
    calib_params_.T_lidar_camera = Eigen::Matrix4d::Identity();

    // Set default point cloud parameters
    pc_params_.min_depth = 0.1f;
    pc_params_.max_depth = 50.0f;

    // 统一的质量控制开关
    pc_params_.enable_quality_filtering = true; // 默认启用质量过滤

    // 质量过滤相关参数
    pc_params_.neighborhood_size = 5;
    pc_params_.depth_discontinuity_threshold = 0.5f;
    pc_params_.gradient_kernel_size = 5;
    pc_params_.depth_ratio_threshold = 2.0f;
    pc_params_.min_neighbor_count = 2;

    // 可视化参数
    pc_params_.save_edge_visualization = true;
    pc_params_.save_gradient_visualization = true;

    // 保留的旧参数（为了兼容性）
    pc_params_.sky_invalid_ratio_threshold = 0.3f;
    pc_params_.sky_detection_kernel_size = 7;

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

    // Set default processing parameters with meaningful paths
    processing_params_.img_sampling_step = 2;
    processing_params_.pcd_path = "REQUIRED: Set path to your .pcd file";
    processing_params_.img_path = "REQUIRED: Set path to your image directory";
    processing_params_.traj_path = "REQUIRED: Set path to your trajectory file";
    processing_params_.output_path = "REQUIRED: Set path to your output directory";
    processing_params_.image_downscale_factor = 1.0;      // 1.0 means no downscaling, 2.0 means half resolution
    processing_params_.use_extrinsic_optimization = true; // Default: enable extrinsic optimization
}

bool Config::loadFromYAML(const std::string &config_file) {
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

    // Directly read path parameters
    // NOTE: These need to be read directly from the root level, not from nested sections
    if (fs["pcd_path"].isString())
        processing_params_.pcd_path = fs["pcd_path"].string();

    if (fs["img_path"].isString())
        processing_params_.img_path = fs["img_path"].string();

    if (fs["traj_path"].isString())
        processing_params_.traj_path = fs["traj_path"].string();

    if (fs["output_path"].isString())
        processing_params_.output_path = fs["output_path"].string();

    // Print loaded paths for debugging
    std::cout << "Loaded paths from config:" << std::endl;
    std::cout << "  - Image path: '" << processing_params_.img_path << "'" << std::endl;
    std::cout << "  - Trajectory path: '" << processing_params_.traj_path << "'" << std::endl;
    std::cout << "  - Point cloud path: '" << processing_params_.pcd_path << "'" << std::endl;
    std::cout << "  - Output path: '" << processing_params_.output_path << "'" << std::endl;

    // Read other parameters from nested sections
    if (fs["point_cloud_params"].isMap()) {
        fs["point_cloud_params"]["min_depth"] >> pc_params_.min_depth;
        fs["point_cloud_params"]["max_depth"] >> pc_params_.max_depth;

        // 读取统一质量控制开关
        if (!fs["point_cloud_params"]["enable_quality_filtering"].empty())
            fs["point_cloud_params"]["enable_quality_filtering"] >> pc_params_.enable_quality_filtering;

        // 读取质量过滤相关参数
        if (!fs["point_cloud_params"]["neighborhood_size"].empty())
            fs["point_cloud_params"]["neighborhood_size"] >> pc_params_.neighborhood_size;
        if (!fs["point_cloud_params"]["depth_discontinuity_threshold"].empty())
            fs["point_cloud_params"]["depth_discontinuity_threshold"] >> pc_params_.depth_discontinuity_threshold;
        if (!fs["point_cloud_params"]["gradient_kernel_size"].empty())
            fs["point_cloud_params"]["gradient_kernel_size"] >> pc_params_.gradient_kernel_size;
        if (!fs["point_cloud_params"]["depth_ratio_threshold"].empty())
            fs["point_cloud_params"]["depth_ratio_threshold"] >> pc_params_.depth_ratio_threshold;
        if (!fs["point_cloud_params"]["min_neighbor_count"].empty())
            fs["point_cloud_params"]["min_neighbor_count"] >> pc_params_.min_neighbor_count;

        // 读取可视化参数
        if (!fs["point_cloud_params"]["save_edge_visualization"].empty())
            fs["point_cloud_params"]["save_edge_visualization"] >> pc_params_.save_edge_visualization;
        if (!fs["point_cloud_params"]["save_gradient_visualization"].empty())
            fs["point_cloud_params"]["save_gradient_visualization"] >> pc_params_.save_gradient_visualization;

        // 读取保留的旧参数（为了兼容性）
        if (!fs["point_cloud_params"]["sky_invalid_ratio_threshold"].empty())
            fs["point_cloud_params"]["sky_invalid_ratio_threshold"] >> pc_params_.sky_invalid_ratio_threshold;
        if (!fs["point_cloud_params"]["sky_detection_kernel_size"].empty())
            fs["point_cloud_params"]["sky_detection_kernel_size"] >> pc_params_.sky_detection_kernel_size;
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
        if (!fs["processing_params"]["img_path"].empty())
            processing_params_.img_path = fs["processing_params"]["img_path"].string();

        if (!fs["processing_params"]["traj_path"].empty())
            processing_params_.traj_path = fs["processing_params"]["traj_path"].string();

        if (!fs["processing_params"]["pcd_path"].empty())
            processing_params_.pcd_path = fs["processing_params"]["pcd_path"].string();

        if (!fs["processing_params"]["output_path"].empty())
            processing_params_.output_path = fs["processing_params"]["output_path"].string();

        if (!fs["processing_params"]["img_sampling_step"].empty())
            fs["processing_params"]["img_sampling_step"] >> processing_params_.img_sampling_step;

        if (!fs["processing_params"]["image_downscale_factor"].empty())
            fs["processing_params"]["image_downscale_factor"] >> processing_params_.image_downscale_factor;

        if (!fs["processing_params"]["use_extrinsic_optimization"].empty())
            fs["processing_params"]["use_extrinsic_optimization"] >> processing_params_.use_extrinsic_optimization;
        if (!fs["processing_params"]["use_time"].empty())
            fs["processing_params"]["use_time"] >> processing_params_.use_time;
        std::cout << processing_params_.use_extrinsic_optimization << std::endl;
    }

    // Final path validation check
    bool paths_valid = true;
    if (processing_params_.img_path.empty()) {
        std::cerr << "Warning: Image path is empty!" << std::endl;
        paths_valid = false;
    }

    if (processing_params_.traj_path.empty()) {
        std::cerr << "Warning: Trajectory path is empty!" << std::endl;
        paths_valid = false;
    }

    if (processing_params_.pcd_path.empty()) {
        std::cerr << "Warning: Point cloud path is empty!" << std::endl;
        paths_valid = false;
    }

    if (processing_params_.output_path.empty()) {
        std::cerr << "Warning: Output path is empty!" << std::endl;
        paths_valid = false;
    }

    if (!paths_valid) {
        std::cerr << "One or more required paths are not properly set in config file: " << config_file << std::endl;
    }

    // 打印质量过滤设置
    std::cout << "Quality filtering configuration:" << std::endl;
    std::cout << "  - Enable quality filtering: " << (pc_params_.enable_quality_filtering ? "true" : "false") << std::endl;
    if (pc_params_.enable_quality_filtering) {
        std::cout << "  - Depth discontinuity threshold: " << pc_params_.depth_discontinuity_threshold << std::endl;
        std::cout << "  - Depth ratio threshold: " << pc_params_.depth_ratio_threshold << std::endl;
        std::cout << "  - Gradient kernel size: " << pc_params_.gradient_kernel_size << std::endl;
        std::cout << "  - Min neighbor count: " << pc_params_.min_neighbor_count << std::endl;
    }

    fs.release();
    return true;
}

void Config::saveToYAML(const std::string &config_file) const {
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
    fs << "point_cloud_params"
       << "{";
    fs << "min_depth" << pc_params_.min_depth;
    fs << "max_depth" << pc_params_.max_depth;

    // 保存统一质量控制开关
    fs << "enable_quality_filtering" << pc_params_.enable_quality_filtering;

    // 保存质量过滤相关参数
    fs << "neighborhood_size" << pc_params_.neighborhood_size;
    fs << "depth_discontinuity_threshold" << pc_params_.depth_discontinuity_threshold;
    fs << "gradient_kernel_size" << pc_params_.gradient_kernel_size;
    fs << "depth_ratio_threshold" << pc_params_.depth_ratio_threshold;
    fs << "min_neighbor_count" << pc_params_.min_neighbor_count;

    // 保存可视化参数
    fs << "save_edge_visualization" << pc_params_.save_edge_visualization;
    fs << "save_gradient_visualization" << pc_params_.save_gradient_visualization;

    // 保存保留的旧参数（为了兼容性）
    fs << "sky_invalid_ratio_threshold" << pc_params_.sky_invalid_ratio_threshold;
    fs << "sky_detection_kernel_size" << pc_params_.sky_detection_kernel_size;
    fs << "}";

    // Save projection parameters
    fs << "projection_params"
       << "{";
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
    fs << "pnp_params"
       << "{";
    fs << "max_rotation_diff" << pnp_params_.max_rotation_diff;
    fs << "max_translation_diff" << pnp_params_.max_translation_diff;
    fs << "reprojection_error" << pnp_params_.reprojection_error;
    fs << "min_inlier_count" << pnp_params_.min_inlier_count;
    fs << "ransac_iterations" << pnp_params_.ransac_iterations;
    fs << "confidence" << pnp_params_.confidence;
    fs << "}";

    // Save output parameters
    fs << "output_params"
       << "{";
    fs << "visualization_tool" << output_params_.visualization_tool;
    fs << "viz_frequency" << output_params_.viz_frequency;
    fs << "}";

    // Save processing parameters
    fs << "processing_params"
       << "{";
    fs << "img_path" << processing_params_.img_path;
    fs << "traj_path" << processing_params_.traj_path;
    fs << "pcd_path" << processing_params_.pcd_path;
    fs << "output_path" << processing_params_.output_path;
    fs << "img_sampling_step" << processing_params_.img_sampling_step;
    fs << "image_downscale_factor" << processing_params_.image_downscale_factor;
    fs << "use_extrinsic_optimization" << processing_params_.use_extrinsic_optimization;
    fs << "}";

    fs.release();
}

} // namespace lvmapping
