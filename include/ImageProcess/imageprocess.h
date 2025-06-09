// created by:  zhiyu zhou 2025/3/2
#ifndef IMG_H
#define IMG_H

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Dense>

class ImageProcess {
public:
    ImageProcess();
    ~ImageProcess();

    /**
     * Project a point cloud to image plane using pinhole model
     * @param cloud Point cloud in camera coordinate system
     * @param isinter Flag for interpolation
     * @return Projected image
     */
    cv::Mat projectPinhole(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, bool isinter);
    
    /**
     * Fill holes in image using fast method
     * @param input Input image with holes
     * @return Filled image
     */
    cv::Mat fillHoles(const cv::Mat& depthImage, int kernelSize);
    cv::Mat fillHolesFast(const cv::Mat& input);

private:
    // Basic camera parameters
    int img_width;
    int img_height;
    float fx, fy, cx, cy; // Camera intrinsics
};

#endif // IMG_H