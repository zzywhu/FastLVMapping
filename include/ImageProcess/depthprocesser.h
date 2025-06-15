#pragma once

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <algorithm>
#include <iostream>

namespace lvmapping {

/**
 * @class Kernels
 * @brief Provides various convolution kernels for morphological operations
 */
class Kernels {
public:
    cv::Mat full_kernel_3;
    cv::Mat full_kernel_5;
    cv::Mat full_kernel_7;
    cv::Mat full_kernel_9;
    cv::Mat full_kernel_31;
    
    /**
     * @brief Constructor initializes basic kernels
     */
    Kernels() {
        // Initialize standard square kernels
        full_kernel_3 = cv::Mat::ones(3, 3, CV_8U);
        full_kernel_5 = cv::Mat::ones(5, 5, CV_8U);
        full_kernel_7 = cv::Mat::ones(7, 7, CV_8U);
        full_kernel_9 = cv::Mat::ones(9, 9, CV_8U);
        full_kernel_31 = cv::Mat::ones(31, 31, CV_8U);
    }
    
    /**
     * @brief Generate a 3x3 cross-shaped kernel
     */
    cv::Mat crossKernel3() const {
        return (cv::Mat_<uint8_t>(3, 3) << 
            0, 1, 0,
            1, 1, 1,
            0, 1, 0);
    }
    
    /**
     * @brief Generate a 5x5 cross-shaped kernel
     */
    cv::Mat crossKernel5() const {
        return (cv::Mat_<uint8_t>(5, 5) << 
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0);
    }
    
    /**
     * @brief Generate a 5x5 diamond-shaped kernel
     */
    cv::Mat diamondKernel5() const {
        return (cv::Mat_<uint8_t>(5, 5) << 
            0, 0, 1, 0, 0,
            0, 1, 1, 1, 0,
            1, 1, 1, 1, 1,
            0, 1, 1, 1, 0,
            0, 0, 1, 0, 0);
    }
    
    /**
     * @brief Generate a 7x7 cross-shaped kernel
     */
    cv::Mat crossKernel7() const {
        return (cv::Mat_<uint8_t>(7, 7) << 
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0);
    }
    
    /**
     * @brief Generate a 7x7 diamond-shaped kernel
     */
    cv::Mat diamondKernel7() const {
        return (cv::Mat_<uint8_t>(7, 7) << 
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 0,
            0, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 0,
            0, 0, 1, 1, 1, 0, 0,
            0, 0, 0, 1, 0, 0, 0);
    }
};

/**
 * @class DepthMapProcessor
 * @brief Core processor for depth map completion
 * 
 * This class applies advanced morphological operations to fill gaps
 * in sparse depth maps from LiDAR projections.
 */
class DepthMapProcessor {
public:
    /**
     * @brief Constructor
     * @param max_depth Maximum depth value for normalization
     */
    DepthMapProcessor(float max_depth = 100.0f) : max_depth_(max_depth) {}
    
    /**
     * @brief Process a sparse depth map to fill gaps
     * @param main_image RGB image (unused, kept for API compatibility)
     * @param depth_map Input sparse depth map
     * @return Processed depth map with gaps filled
     */
    cv::Mat createMap(const cv::Mat& main_image, const cv::Mat& depth_map) {
        // Convert depth map to floating point
        cv::Mat depths = depth_map.clone();
        depths.convertTo(depths, CV_32F);
        
        // Apply multi-stage processing pipeline
        cv::Mat processed_depths = processDepthMap(depths);
        
        return processed_depths;
    }

private:
    float max_depth_;
    Kernels kernels_;
    
    /**
     * @brief Main depth map processing pipeline
     * @param depths Input depth map
     * @return Processed depth map
     */
    cv::Mat processDepthMap(const cv::Mat& depths) {
        // 1. Segment depth map by ranges
        cv::Mat near_mask, med_mask, far_mask;
        segmentDepthRanges(depths, near_mask, med_mask, far_mask);
        
        // 2. Apply range-adaptive morphology
        cv::Mat dilated_depths = applyRangeMorphology(depths, near_mask, med_mask, far_mask);
        
        // 3. Fill small holes
        cv::Mat closed_depths;
        cv::morphologyEx(dilated_depths, closed_depths, cv::MORPH_CLOSE, kernels_.full_kernel_5);
        
        // 4. Apply median filtering to denoise
        cv::Mat filtered_depths = applyMedianFilter(closed_depths);
        
        // 5. Fill top regions
        cv::Mat extended_depths = fillTopRegions(filtered_depths);
        
        // 6. Final smoothing
        cv::Mat smoothed_depths = applyFinalSmoothing(extended_depths);
        
        return smoothed_depths;
    }
    
    /**
     * @brief Segment depth map into near, medium, and far ranges
     */
    void segmentDepthRanges(const cv::Mat& depths, cv::Mat& near_mask, cv::Mat& med_mask, cv::Mat& far_mask) {
        // Create range masks (0.1-15m, 15-30m, >30m)
        cv::inRange(depths, 0.1f, 15.0f, near_mask);
        cv::inRange(depths, 15.0f, 30.0f, med_mask);
        cv::threshold(depths, far_mask, 30.0f, 255, cv::THRESH_BINARY);
        
        // Convert to floating point masks (0.0 or 1.0)
        near_mask.convertTo(near_mask, CV_32F, 1.0/255.0);
        med_mask.convertTo(med_mask, CV_32F, 1.0/255.0);
        far_mask.convertTo(far_mask, CV_32F, 1.0/255.0);
    }
    
    /**
     * @brief Apply different morphological operations based on depth range
     */
    cv::Mat applyRangeMorphology(const cv::Mat& depths, 
                                const cv::Mat& near_mask, 
                                const cv::Mat& med_mask, 
                                const cv::Mat& far_mask) {
        // Extract depth ranges
        cv::Mat near_depths = depths.mul(near_mask);
        cv::Mat med_depths = depths.mul(med_mask);
        cv::Mat far_depths = depths.mul(far_mask);
        
        // Apply different dilation kernels based on depth range
        cv::Mat dilated_near, dilated_med, dilated_far;
        cv::dilate(near_depths, dilated_near, kernels_.diamondKernel7());
        cv::dilate(med_depths, dilated_med, kernels_.diamondKernel5());
        cv::dilate(far_depths, dilated_far, kernels_.crossKernel3());
        
        // Create masks for the dilated regions
        cv::Mat near_dilated_mask, med_dilated_mask, far_dilated_mask;
        cv::threshold(dilated_near, near_dilated_mask, 0.1f, 1.0f, cv::THRESH_BINARY);
        cv::threshold(dilated_med, med_dilated_mask, 0.1f, 1.0f, cv::THRESH_BINARY);
        cv::threshold(dilated_far, far_dilated_mask, 0.1f, 1.0f, cv::THRESH_BINARY);
        
        // Merge results with priority (near > medium > far)
        cv::Mat result = depths.clone();
        
        // Apply far depth first (lowest priority)
        result = result.mul(1.0 - far_dilated_mask) + dilated_far;
        
        // Apply medium depth (medium priority)
        result = result.mul(1.0 - med_dilated_mask) + dilated_med;
        
        // Apply near depth (highest priority)
        result = result.mul(1.0 - near_dilated_mask) + dilated_near;
        
        return result;
    }
    
    /**
     * @brief Apply median filtering to valid depths
     */
    cv::Mat applyMedianFilter(const cv::Mat& depths) {
        cv::Mat result = depths.clone();
        cv::Mat valid_mask;
        cv::threshold(depths, valid_mask, 0.1f, 1.0f, cv::THRESH_BINARY);
        
        // Apply median blur
        cv::Mat blurred;
        cv::medianBlur(depths, blurred, 5);
        
        // Only update valid pixels
        result = result.mul(1.0 - valid_mask) + blurred.mul(valid_mask);
        
        return result;
    }
    
    /**
     * @brief Fill top regions of the depth map
     */
    cv::Mat fillTopRegions(const cv::Mat& depths) {
        cv::Mat result = depths.clone();
        
        // Create top mask
        cv::Mat top_mask = cv::Mat::ones(depths.size(), CV_8U);
        
        // Find top rows with no depth
        for (int x = 0; x < depths.cols; x++) {
            int top_row = 0;
            while (top_row < depths.rows && depths.at<float>(top_row, x) <= 0.1f) {
                top_row++;
            }
            
            // Mark these as part of the top region
            for (int y = 0; y < top_row; y++) {
                top_mask.at<uchar>(y, x) = 0;
            }
        }
        
        // Find empty regions
        cv::Mat valid_mask, empty_mask;
        cv::threshold(depths, valid_mask, 0.1f, 255, cv::THRESH_BINARY);
        cv::bitwise_not(valid_mask, empty_mask);
        cv::bitwise_and(empty_mask, top_mask, empty_mask);
        
        // Dilate to fill empty regions
        cv::Mat dilated;
        cv::dilate(depths, dilated, kernels_.full_kernel_7);
        
        // Update empty regions with dilated values
        for (int y = 0; y < result.rows; y++) {
            for (int x = 0; x < result.cols; x++) {
                if (empty_mask.at<uchar>(y, x) > 0) {
                    result.at<float>(y, x) = dilated.at<float>(y, x);
                }
            }
        }
        
        return result;
    }
    
    /**
     * @brief Apply final smoothing and gap filling
     */
    cv::Mat applyFinalSmoothing(const cv::Mat& depths) {
        cv::Mat result = depths.clone();
        cv::Mat valid_mask, empty_mask, top_mask;
        
        // Create top mask (same as in fillTopRegions)
        top_mask = cv::Mat::ones(depths.size(), CV_8U);
        for (int x = 0; x < depths.cols; x++) {
            int top_row = 0;
            while (top_row < depths.rows && depths.at<float>(top_row, x) <= 0.1f) {
                top_row++;
            }
            for (int y = 0; y < top_row; y++) {
                top_mask.at<uchar>(y, x) = 0;
            }
        }
        
        // Multiple iterations of fill and smooth
        for (int i = 0; i < 3; i++) {  // Reduced to 3 iterations for efficiency
            // Create empty pixel mask
            cv::threshold(result, valid_mask, 0.1f, 255, cv::THRESH_BINARY);
            cv::bitwise_not(valid_mask, empty_mask);
            cv::bitwise_and(empty_mask, top_mask, empty_mask);
            
            // Fill with dilated values
            cv::Mat dilated;
            cv::dilate(result, dilated, kernels_.full_kernel_9);  // Reduced kernel size
            
            for (int y = 0; y < result.rows; y++) {
                for (int x = 0; x < result.cols; x++) {
                    if (empty_mask.at<uchar>(y, x) > 0) {
                        result.at<float>(y, x) = dilated.at<float>(y, x);
                    }
                }
            }
        }
        
        // Final median blur
        cv::threshold(result, valid_mask, 0.1f, 255, cv::THRESH_BINARY);
        cv::Mat blurred;
        cv::medianBlur(result, blurred, 5);
        
        for (int y = 0; y < result.rows; y++) {
            for (int x = 0; x < result.cols; x++) {
                if (valid_mask.at<uchar>(y, x) > 0) {
                    result.at<float>(y, x) = blurred.at<float>(y, x);
                }
            }
        }
        
        // Final gaussian blur
        cv::GaussianBlur(result, blurred, cv::Size(3, 3), 0);
        
        for (int y = 0; y < result.rows; y++) {
            for (int x = 0; x < result.cols; x++) {
                if (valid_mask.at<uchar>(y, x) > 0) {
                    result.at<float>(y, x) = blurred.at<float>(y, x);
                }
            }
        }
        
        return result;
    }
};

} // namespace lvmapping