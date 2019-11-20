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
        void optimize(const Eigen::Matrix<double, 3, Eigen::Dynamic>& data_cloud,
                const Eigen::VectorXi& data_part_labels,
                int num_parts, const int* part_map,
                int icp_iters = 1, int num_threads = 4);

        /** Rotation representation size */
        static const int ROT_SIZE = 4;

        /** Optimization parameter r */
        std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > r;

        /** Cost function component weights */
        double betaPose = 0.1, betaShape = 1.0, betaJoints = 0.2;

        /** NN matching step size; only matches nearest neighbors every 
         *  x points to speed up optimization (sort of hack) */
        int nnStep = 20;

        /** The avatar we are optimizing */
        Avatar& ava;
        /** Camera intrinsics */
        const CameraIntrin& intrin;
        /** Input image size */
        cv::Size imageSize;
    };
}
