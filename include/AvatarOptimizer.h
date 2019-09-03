#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

namespace ark {
    class Avatar;
    /** Optimization tools */
    class AvatarOptimizer {
    public:
        AvatarOptimizer(Avatar& ava);

        /** Begin optimization on the target data cloud */
        void optimize(const Eigen::Matrix<double, 3, Eigen::Dynamic>& data_cloud, int icp_iters = 1);

        Avatar& ava;

        /** Rotation representation size */
        static const int ROT_SIZE = 4;

        /** Optimization parameter r */
        std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > r;
    };
}
