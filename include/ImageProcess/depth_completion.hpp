#include <opencv2/opencv.hpp>
#include <vector>

namespace lvmapping {

/**
 * @class DepthCompletion 
 * @brief Efficient depth completion algorithm
 */
class DepthCompletion {
public:
    /**
     * @brief Process a sparse depth map to fill in gaps
     * @param lidar_depth Input sparse depth map
     * @param img_width Image width (optional)
     * @param img_height Image height (optional)
     * @return Dense depth map
     */
    static cv::Mat process(const cv::Mat& lidar_depth, int img_width = 0, int img_height = 0) {
        // Initialize kernels with efficient sizes
        initializeKernels();
        
        // Clone input depth map
        cv::Mat depth = lidar_depth.clone();

        // Remove very small values and normalize
        cv::Mat mask = (depth > 0.1);
        
        // Optional depth inversion (from far=bright to near=bright)
        cv::subtract(100.0, depth, depth, mask);
        
        // Apply fast multi-step processing
        applyMorphologicalOperations(depth);
        
        // Invert back to original depth representation
        mask = (depth > 0.1);
        cv::subtract(100.0, depth, depth, mask);
        
        return depth;
    }

private:
    // Kernels
    static cv::Mat full_kernel3x3_;
    static cv::Mat full_kernel5x5_;
    static cv::Mat diamond_kernel5x5_;
    
    /**
     * @brief Initialize optimized morphological kernels
     */
    static void initializeKernels() {
        static bool initialized = false;
        
        if (!initialized) {
            // Create kernels only once
            full_kernel3x3_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            full_kernel5x5_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            
            // Create optimized diamond kernel
            diamond_kernel5x5_ = (cv::Mat_<uint8_t>(5, 5) << 
                0, 0, 1, 0, 0,
                0, 1, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 1, 1, 0,
                0, 0, 1, 0, 0);
                
            initialized = true;
        }
    }
    
    /**
     * @brief Apply efficient morphological operations
     * @param depth Depth map to process (in-place)
     */
    static void applyMorphologicalOperations(cv::Mat& depth) {
        cv::Mat mask, dilated_map;
        
        // 1. Initial dilation with small optimized kernel
        cv::dilate(depth, depth, diamond_kernel5x5_);
        
        // 2. Fill small holes with morphological closing
        cv::morphologyEx(depth, depth, cv::MORPH_CLOSE, full_kernel3x3_);
        
        // 3. Fill small gaps
        mask = (depth < 0.1);
        cv::dilate(depth, dilated_map, full_kernel3x3_);
        dilated_map.copyTo(depth, mask);
        
        // 4. Fill larger gaps with targeted dilation
        mask = (depth < 0.1);
        cv::dilate(depth, dilated_map, full_kernel5x5_);
        dilated_map.copyTo(depth, mask);
        
        // 5. Denoise with median filter
        cv::Mat filtered;
        cv::medianBlur(depth, filtered, 3);
        
        mask = (depth > 0.1);
        filtered.copyTo(depth, mask);
        
        // 6. Edge-preserving smoothing with small Gaussian
        cv::GaussianBlur(depth, filtered, cv::Size(3, 3), 0);
        filtered.copyTo(depth, mask);
    }
};

// Initialize static members
cv::Mat DepthCompletion::full_kernel3x3_;
cv::Mat DepthCompletion::full_kernel5x5_;
cv::Mat DepthCompletion::diamond_kernel5x5_;

/**
 * @brief Legacy function maintained for backwards compatibility
 */
cv::Mat depth_completion(const cv::Mat& lidar_depth, int img_width, int img_height) {
    return DepthCompletion::process(lidar_depth, img_width, img_height);
}

} // namespace lvmapping