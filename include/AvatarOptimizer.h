#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include "Calibration.h"

namespace ark {
    class Avatar;
    /** Optimize avatar to fit a point cloud */
    class AvatarOptimizer {
    public:
        /** Construct avatar optimizer for given avatar, with avatar intrinsics and image size */
        AvatarOptimizer(Avatar& ava, const CameraIntrin& intrin, const cv::Size& image_size);

        /** Begin full optimization on the target data cloud */
        void optimize(const Eigen::Matrix<double, 3, Eigen::Dynamic>& data_cloud, int icp_iters = 1, int num_threads = 4);

        /** Begin align optimization to fit joints to target joint positions */
        void align(const Eigen::Matrix<double, 3, Eigen::Dynamic>& smpl_joints, int icp_iters = 1, int num_threads = 4);

        /** Rotation representation size */
        static const int ROT_SIZE = 4;

        /** Optimization parameter r */
        std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > r;

        /** Cost function component weights */
        double betaPose = 0.1, betaShape = 1.0, betaJoints = 0.2;

        /** The avatar we are optimizing */
        Avatar& ava;
        /** Camera intrinsics */
        const CameraIntrin& intrin;
        /** Input image size */
        const cv::Size& imageSize;
    };
}
