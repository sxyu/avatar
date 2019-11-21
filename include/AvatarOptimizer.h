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
        AvatarOptimizer(Avatar& ava, const CameraIntrin& intrin, const cv::Size& image_size, int num_parts, const int* part_map);

        /** Begin full optimization on the target data cloud */
        void optimize(const Eigen::Matrix<double, 3, Eigen::Dynamic>& data_cloud,
                const Eigen::VectorXi& data_part_labels,
                int icp_iters = 1, int num_threads = 4);

        /** Rotation representation size */
        static const int ROT_SIZE = 4;

        /** Optimization parameter r */
        std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > r;

        /** Cost function component weights */
        double betaPose = 0.1, betaShape = 1.0;

        /** NN matching step size; only matches nearest neighbors
         *  every x points to speed up optimization
         *  (sort of hack; only applied in forward NN matching mode) */
        int nnStep = 20;

        /** Whether to elimiate occluded points before NN matching */
        bool enableOcclusion = true;

        /** The avatar we are optimizing */
        Avatar& ava;
        /** Camera intrinsics */
        const CameraIntrin& intrin;
        /** Input image size */
        cv::Size imageSize;

        /** Number of body parts for random tree inference */
        int numParts;

        /** Mapping from assigned joint to body part
         * (as defined for the RTree used, was given during training) */
        const int * partMap;
    private:
        /** Internal precomputed values for avatar model body part
         *  sizes/counts which are constant across optimize calls */
        std::vector<Eigen::VectorXi> modelPartIndices;
        Eigen::VectorXi modelPartLabelCounts;
        std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic> > modelPartClouds;

    };
}
