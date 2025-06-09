#include<iostream>
#include"calib.h"
#include"ImageProcess/imageprocess.h"
#include<string>
#include<opencv2/opencv.hpp>
#include<cstdlib>   // For system() function

// Function to check if a command is available
bool isCommandAvailable(const std::string& command) {
    std::string test_cmd = "which " + command + " > /dev/null 2>&1";
    return system(test_cmd.c_str()) == 0;
}

// Function to suggest visualization tools
void suggestVisualizationTools(const std::string& pcd_path) {
    std::cout << "Point cloud visualization options:" << std::endl;
    
    if (isCommandAvailable("pcl_viewer")) {
        std::cout << "Option 1: Use PCL Viewer: pcl_viewer " << pcd_path << std::endl;
    } else {
        std::cout << "PCL Viewer not found. You can install it with:" << std::endl;
        std::cout << "sudo apt install pcl-tools" << std::endl;
    }
    
    if (isCommandAvailable("cloudcompare")) {
        std::cout << "Option 2: Use CloudCompare: cloudcompare " << pcd_path << std::endl;
    } else {
        std::cout << "CloudCompare not found. You can install it with:" << std::endl;
        std::cout << "sudo apt install cloudcompare" << std::endl;
    }
    
    if (isCommandAvailable("meshlab")) {
        std::cout << "Option 3: Use MeshLab: meshlab " << pcd_path << std::endl;
    } else {
        std::cout << "MeshLab not found. You can install it with:" << std::endl;
        std::cout << "sudo apt install meshlab" << std::endl;
    }
    
    std::cout << "You can open the colored point cloud with any PCD viewer application." << std::endl;
}

int main()
{
    std::string pcdpath; 
    std::string imgpath; 
    std::string trajpath; 
    std::string outpath;
    std::string calib_config_file("/home/zzy/SensorCalibration/FastLVMapping/config/param.yaml");
    cv::FileStorage fSettings(calib_config_file, cv::FileStorage::READ);
    
    pcdpath = fSettings["pcd_path"].string();
    imgpath = fSettings["img_path"].string();
    trajpath = fSettings["traj_path"].string();
    outpath = fSettings["output_path"].string();

    // Initialize calibration object
    CalibProcessor calib;
    
    std::cout << "Starting image processing and projection..." << std::endl;
    std::cout << "Point cloud path: " << pcdpath << std::endl;
    std::cout << "Image folder path: " << imgpath << std::endl;
    std::cout << "Trajectory file path: " << trajpath << std::endl;
    std::cout << "Output directory: " << outpath<< std::endl;
    
    // Check visualization tools before processing
    std::string colormap_dir = outpath + "/colormap";
    std::string colored_pcd_path = colormap_dir + "/colored_pointcloud.pcd";
    
    // Provide alternative viewer options to pcl_viewer
    std::string viewer_command = "pcl_viewer";  // Default viewer
    
    if (!isCommandAvailable(viewer_command)) {
        std::cout << "Warning: pcl_viewer not found. Checking for alternatives..." << std::endl;
        
        // Check for alternatives
        if (isCommandAvailable("cloudcompare")) {
            viewer_command = "cloudcompare";
        } else if (isCommandAvailable("meshlab")) {
            viewer_command = "meshlab";
        } else {
            std::cout << "No suitable point cloud viewer found. You'll need to install one." << std::endl;
            std::cout << "Processing will continue but without real-time visualization." << std::endl;
            viewer_command = "";
        }
    }
    
    // Pass viewer information to CalibProcessor
    calib.setVisualizationTool(viewer_command);
    
    // Process images and project point cloud
    bool success = calib.processImagesAndPointCloud(imgpath, trajpath, pcdpath, outpath);
    
    if (success) {
        std::cout << "Successfully processed all images and created projections." << std::endl;
        std::cout << "Results saved to: " << outpath << std::endl;
        
        // Suggest visualization tools for the final result
        suggestVisualizationTools(colored_pcd_path);
    } else {
        std::cerr << "Error during image processing and projection." << std::endl;
        return -1;
    }

    return 0;
}
