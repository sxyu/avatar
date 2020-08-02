#pragma once

#include <utility>
#include <algorithm>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace ark {

/** Hand-written faster function to load a saved PCL point cloud directly
 *  into an Eigen vector, where points are stored: x1 y1 z1 x2 y2 z2 ...
 *  The reason we flatten the cloud instead of using a matrix is to make it
 * easier to add in shape keys, which would otherwise need to be tensors */
Eigen::VectorXd loadPCDToPointVectorFast(const std::string& path);

/** Spherical to rectangular coords */
void fromSpherical(double rho, double theta, double phi, Eigen::Vector3d& out);

/** Paint projected triangle on depth map using barycentric linear interp */
template <class T>
void paintTriangleBary(cv::Mat& output, const cv::Size& image_size,
                       const std::vector<cv::Point2f>& projected,
                       const cv::Vec3i& face, const float* zvec,
                       float maxz = 255.0f);

/** Paint projected triangle on part mask (CV_8U) using nearest neighbors interp
 */
void paintPartsTriangleNN(
    cv::Mat& output_assigned_joint_mask, const cv::Size& image_size,
    const std::vector<cv::Point2f>& projected,
    const std::vector<std::vector<std::pair<double, int>>>& assigned_joint,
    const cv::Vec3i& face, const std::vector<int>& part_map);

/** Paint projected triangle on image to single color (based on template type)
 */
template <class T>
void paintTriangleSingleColor(cv::Mat& output_image, const cv::Size& image_size,
                              const std::vector<cv::Point2f>& projected,
                              const cv::Vec3i& face, T color);
}  // namespace ark
