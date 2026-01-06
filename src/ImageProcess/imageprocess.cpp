#include "ImageProcess/imageprocess.h"
#include "config.h"
#include <limits>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// ImageProcess implementation
ImageProcess::ImageProcess() :
    img_width(0), img_height(0), fx(0), fy(0), cx(0), cy(0) {
}

ImageProcess::~ImageProcess() {
}

cv::Mat ImageProcess::projectPinhole(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, bool isinter) {
    // 设置图像尺寸
    const int IMG_HEIGHT = 800; // 图像高度
    const int IMG_WIDTH = 800;  // 图像宽度

    // 设置相机内参
    const float fx = 400.0f;            // 焦距x
    const float fy = 400.0f;            // 焦距y
    const float cx = IMG_WIDTH / 2.0f;  // 光心x坐标
    const float cy = IMG_HEIGHT / 2.0f; // 光心y坐标

    // 创建强度图，初始化为0
    cv::Mat intensityImage = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32F);

    // 如果点云为空，返回空图像
    if (cloud->size() == 0) {
        intensityImage = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U);
        return intensityImage;
    }

    // 创建深度图，用于处理遮挡
    cv::Mat depthMap = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32F);
    depthMap.setTo(std::numeric_limits<float>::max()); // 初始化为最大值

    // 投影点云到图像平面
    for (const auto &p : cloud->points) {
        // 只处理相机前方的点（假设相机朝向-x方向）
        if (p.z < 0) {
            continue;
        }

        // 计算投影点坐标
        float x = p.x;
        float y = p.y;
        float z = p.z;

        // 针孔投影
        float depth = z; // 深度值为x轴距离

        // 如果深度为0或负数，跳过
        if (depth <= 0) {
            continue;
        }

        // 计算像素坐标
        float u = (fx * x / z) + cx;
        float v = (fy * y / z) + cy;

        // 检查像素坐标是否在图像范围内
        if (u >= 0 && u < IMG_WIDTH && v >= 0 && v < IMG_HEIGHT) {
            // 强制转换为整数坐标
            int ui = static_cast<int>(u);
            int vi = static_cast<int>(v);

            // 处理遮挡（近处的点覆盖远处的点）
            if (depth < depthMap.at<float>(vi, ui)) {
                depthMap.at<float>(vi, ui) = depth;
                intensityImage.at<float>(vi, ui) = p.intensity;
            }
        }
    }

    // 归一化到 [0, 255]
    double minVal, maxVal;
    cv::minMaxLoc(intensityImage, &minVal, &maxVal);

    // 避免除以零
    if (maxVal > minVal) {
        intensityImage.convertTo(intensityImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    } else {
        intensityImage.convertTo(intensityImage, CV_8U);
    }

    // 直方图均衡化
    cv::equalizeHist(intensityImage, intensityImage);

    // 伽马校正增强对比度
    cv::Mat gammaImage;
    intensityImage.convertTo(gammaImage, CV_32F, 1.0 / 255.0);
    cv::pow(gammaImage, 0.5, gammaImage); // γ = 0.5
    gammaImage.convertTo(intensityImage, CV_8U, 255);
    cv::Mat finalresult;
    if (isinter) {
        finalresult = fillHoles(intensityImage, 1);
    } else {
        finalresult = intensityImage;
    }

    return finalresult;
}

cv::Mat ImageProcess::fillHoles(const cv::Mat &depthImage, int kernelSize) {
    // Copy the input image to avoid modifying it
    cv::Mat filledImage = depthImage.clone();

    // Create a mask for holes (zero values)
    cv::Mat mask = cv::Mat::zeros(depthImage.size(), CV_8U);
    for (int y = 0; y < depthImage.rows; y++) {
        for (int x = 0; x < depthImage.cols; x++) {
            float depth = depthImage.at<float>(y, x);
            if (depth <= 0 || std::isnan(depth)) {
                mask.at<uchar>(y, x) = 255; // Mark holes
            }
        }
    }

    // Apply median filter to fill holes
    cv::Mat medianFiltered;
    cv::medianBlur(filledImage, medianFiltered, kernelSize);

    // Copy median values only to hole positions
    medianFiltered.copyTo(filledImage, mask);

    return filledImage;
}

// Simple method to fill holes fast using neighboring pixels
cv::Mat ImageProcess::fillHolesFast(const cv::Mat &input) {
    if (input.empty()) {
        return input.clone();
    }

    // Ensure input image is 8-bit single channel
    cv::Mat processImage;
    if (input.type() != CV_8U) {
        input.convertTo(processImage, CV_8U);
    } else {
        processImage = input.clone();
    }

    // Create output image
    cv::Mat result = processImage.clone();

    // Identify hole locations
    cv::Mat mask = (processImage == 0);
    std::vector<cv::Point> holePoints;
    cv::findNonZero(mask, holePoints);

    // If no holes, return directly
    if (holePoints.empty()) {
        return result;
    }

    // Fixed search radius
    const int radius = 2;

// Process each hole point once
#pragma omp parallel for
    for (int idx = 0; idx < holePoints.size(); idx++) {
        const cv::Point &point = holePoints[idx];
        int x = point.x;
        int y = point.y;

        // Collect values from surrounding non-hole points
        int totalValue = 0;
        int count = 0;

        // Search for valid points in neighborhood
        for (int j = -radius; j <= radius; j++) {
            for (int i = -radius; i <= radius; i++) {
                int nx = x + i;
                int ny = y + j;

                // Check boundaries
                if (nx >= 0 && nx < processImage.cols && ny >= 0 && ny < processImage.rows) {
                    uchar value = processImage.at<uchar>(ny, nx);
                    if (value > 0) { // Non-hole point
                        totalValue += value;
                        count++;
                    }
                }
            }
        }

        // Fill with average if valid points found, otherwise use default
        if (count > 0) {
            result.at<uchar>(y, x) = totalValue / count;
        } else {
            result.at<uchar>(y, x) = 0;
        }
    }

    return result;
}

cv::Mat ImageProcess::projectPinhole(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, bool colorize) {
    // Call the new method and return just the image
    return projectPinholeWithIndices(cloud, colorize).first;
}

std::pair<cv::Mat, cv::Mat> ImageProcess::projectPinholeWithIndices(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, bool isinter) {
    // Get projection parameters from config
    const lvmapping::Config &config = lvmapping::Config::getInstance();
    const auto &proj_params = config.projectionParams();
    cv::Mat resize_matrix_ = config.cameraParams().resize_camera_matrix.clone();
    // Set image dimensions from config
    const int IMG_HEIGHT = resize_matrix_.at<float>(1, 2) * 2;
    const int IMG_WIDTH = resize_matrix_.at<float>(0, 2) * 2;

    // Set camera intrinsics from config
    const float fx = static_cast<float>(resize_matrix_.at<float>(0, 0));
    const float fy = static_cast<float>(resize_matrix_.at<float>(1, 1));
    const float cx = static_cast<float>(resize_matrix_.at<float>(0, 2));
    const float cy = static_cast<float>(resize_matrix_.at<float>(1, 2));

    // Create intensity image, initialized to 0
    cv::Mat intensityImage = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32F);

    // Create index map, initialized to -1 (indicating no corresponding point cloud point)
    cv::Mat indexMap = cv::Mat(IMG_HEIGHT, IMG_WIDTH, CV_32S, cv::Scalar(-1));

    // If point cloud is empty, return empty image
    if (cloud->size() == 0) {
        cv::Mat emptyImage = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U);
        return {emptyImage, indexMap};
    }

    // Create depth map for handling occlusions
    cv::Mat depthMap = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32F);
    depthMap.setTo(std::numeric_limits<float>::max()); // Initialize to maximum value

    // Project point cloud onto image plane
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        const auto &p = cloud->points[i];

        // Only process points in front of the camera
        if (p.z < 0) {
            continue;
        }

        // Calculate projected point coordinates
        float x = p.x;
        float y = p.y;
        float z = p.z;

        // Pinhole projection
        float depth = z; // Depth value is the z-axis distance

        // Skip if depth is 0 or negative
        if (depth <= 0) {
            continue;
        }

        // Calculate pixel coordinates
        float u = (fx * x / z) + cx;
        float v = (fy * y / z) + cy;

        // Check if pixel coordinates are within image bounds
        if (u >= 0 && u < IMG_WIDTH && v >= 0 && v < IMG_HEIGHT) {
            // Force convert to integer coordinates
            int ui = static_cast<int>(u);
            int vi = static_cast<int>(v);

            // Handle occlusions (closer points override further points)
            if (depth < depthMap.at<float>(vi, ui)) {
                depthMap.at<float>(vi, ui) = depth;
                intensityImage.at<float>(vi, ui) = p.intensity;
                indexMap.at<int>(vi, ui) = static_cast<int>(i); // Record point cloud index
            }
        }
    }

    // 归一化到 [0, 255]
    double minVal, maxVal;
    cv::minMaxLoc(intensityImage, &minVal, &maxVal);

    cv::Mat normalizedImage;
    // 避免除以零
    if (maxVal > minVal) {
        intensityImage.convertTo(normalizedImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    } else {
        intensityImage.convertTo(normalizedImage, CV_8U);
    }

    cv::imwrite("/home/zzy/SensorCalibration/EasyColor/src/FastLVMapping/testdata/intensity_image.png", normalizedImage); // 保存强度图像用于调试

    // 直方图均衡化
    cv::equalizeHist(normalizedImage, normalizedImage);
    cv::imwrite("/home/zzy/SensorCalibration/EasyColor/src/FastLVMapping/testdata/normalized_image.png", normalizedImage); // 保存归一化图像用于调试

    // 伽马校正增强对比度
    cv::Mat gammaImage;
    normalizedImage.convertTo(gammaImage, CV_32F, 1.0 / 255.0);
    cv::pow(gammaImage, 0.5, gammaImage); // γ = 0.5
    gammaImage.convertTo(normalizedImage, CV_8U, 255);

    cv::imwrite("/home/zzy/SensorCalibration/EasyColor/src/FastLVMapping/testdata/gamma_corrected_image.png", normalizedImage); // 保存伽马校正图像用于调试

    // 将灰度图转换为彩色图像
    cv::Mat colorImage;
    cv::cvtColor(normalizedImage, colorImage, cv::COLOR_GRAY2BGR);

    cv::Mat finalImage;
    if (isinter) {
        // 注意：fillHoles需要相应的修改来处理彩色图像
        cv::Mat filledGray = fillHoles(normalizedImage, 1);
        cv::cvtColor(filledGray, finalImage, cv::COLOR_GRAY2BGR);
    } else {
        finalImage = colorImage;
    }
    cv::imwrite("/home/zzy/SensorCalibration/EasyColor/src/FastLVMapping/testdata/final_image.png", finalImage); // 保存最终图像用于调试

    return {finalImage, indexMap};
}