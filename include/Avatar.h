#pragma once
#include "Version.h"
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace ark {
    class HumanDetector;
    struct HumanAvatarUKFModel;

    typedef Eigen::Matrix<float, 3, Eigen::Dynamic> CloudType;
    typedef Eigen::Matrix<float, 2, Eigen::Dynamic> Cloud2DType;
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

    /** Names for the various skeletal joints in the SMPL model (does not work for other models) */
    namespace SmplJointType {
        enum {
            // TODO: delegate to a skeleton instead

            // BFS Order (topologically sorted)
            ROOT_PELVIS = 0, L_HIP, R_HIP, SPINE1, L_KNEE, R_KNEE, SPINE2, L_ANKLE,
            R_ANKLE, SPINE3, L_FOOT, R_FOOT, NECK, L_COLLAR, R_COLLAR, HEAD, L_SHOULDER,
            R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HAND, R_HAND,

            _COUNT
        };
    }

    /** Represents a generic humanoid avatar */
    class HumanAvatar {
    public:
        /** Create an avatar from the model information in 'model_dir'
         *  @param model_dir path to directory containing model files
         */
        explicit HumanAvatar(const std::string & model_dir);

        /** Update the avatar's shape and pose */
        void update();

        /** Get number of joints */
        inline int numJoints() const { return jointRegressor.cols(); }

        /** Get number of skin points */
        inline int numPoints() const { return jointRegressor.rows(); }

        /** Get number of shape keys */
        inline int numShapeKeys() const { return keyClouds.cols(); }

        /** Compute the avatar's SMPL pose parameters (Rodrigues angles) */
        Eigen::VectorXf smplParams() const;

        /** Get PCL point cloud of Avatar's points */
        pcl::PointCloud<pcl::PointXYZ>::Ptr getCloud() const;

        /** Shape-key (aka. blend shape) weights */
        Eigen::VectorXf w;

        /** Root position */
        Eigen::Vector3f p;

        /** The rotations, stored as 3x3 rotation matrices */
        std::vector<Eigen::Matrix3f> r;

        /** Current point cloud with pose and shape keys both applied (3, num points) */
        CloudType cloud;

        /** Current joint positions (3, num joints) */
        CloudType jointPos;

        /** Parent joint index of each joint */
        Eigen::VectorXi parent;

        /** The directory the avatar's model was imported from */
        const std::string MODEL_DIR;

    private:
        /** Base point cloud with positions of each skin point from data file (3 * num points) */
        Eigen::VectorXf baseCloud;

        /** Shape key (blendshape) data (3*num points, num keys),
         *  each column is vectorized matrix of points x1 y1 z1 x2 y2 z2 ... */
        MatrixType keyClouds;

        /** Joint regressor (num points, num joints) */
        Eigen::SparseMatrix<float> jointRegressor;

        /** Assignment weights: weights for point-joints assignments.
         *  Columns are recorded in 'assignment' indices,
         *  see assignStarts below. (num assignments total, num points) */
        Eigen::SparseMatrix<float> assignWeights;

        /** Points for each assigned point (3, num assignments total) */
        CloudType assignVecs;

        /** Start index of each joint's assigned points
         *  as in cols of assignVecs and rows of assignWeights (num joints + 1);
         *  terminated with num assignments total */
        Eigen::VectorXi assignStarts;

        /** List of points assigned to each joint with weight*/
        std::vector<std::vector<std::pair<int, float> > > assignedPoints;
    };
}
