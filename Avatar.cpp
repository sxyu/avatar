#include "Avatar.h"
#include <boost/filesystem.hpp>
#include <boost/smart_ptr.hpp>
#include <pcl/io/pcd_io.h>

namespace {
    Eigen::VectorXf loadPCDToPointVector(const std::string& path) {
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::io::loadPCDFile<pcl::PointXYZ>(path, cloud);
        Eigen::VectorXf result(cloud.size() * 3);
        for (size_t i = 0 ; i < cloud.size(); ++i) {
            result.segment<3>(i*3) = cloud[i].getVector3fMap();
        }
        //Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > matMap(result.data(), cloud.size(), 3);
        //matMap = cloud.getMatrixXfMap();
        return result;
    }
}

namespace ark {
    HumanAvatar::HumanAvatar(const std::string & model_dir) : MODEL_DIR(model_dir) {
        using PclCloudType = pcl::PointCloud<pcl::PointXYZ>;

        auto humanPCBase = std::unique_ptr<PclCloudType>(new PclCloudType());
        auto humanPCTransformed = boost::make_shared<PclCloudType>();

        using namespace boost::filesystem;
        path skelPath = path(model_dir) / "skeleton.txt";
        path jrPath = path(model_dir) / "joint_regressor.txt";

        baseCloud = loadPCDToPointVector((path(model_dir) / "model.pcd").string());

        int nJoints, nPoints;
        // Read skeleton file
        std::ifstream skel(skelPath.string());
        skel >> nJoints >> nPoints;

        // Assume joints are given in topologically sorted order
        parent.resize(nJoints);
        for (int i = 0; i < nJoints; ++i) {
            int id;
            std::string _name; float _x, _y, _z; // throw away

            skel >> id;
            skel >> parent[id];
            skel >> _name >> _x >> _y >> _z;
        }

		int ii = 0;
        if (!skel) {
            std::cerr << "ERROR: Invalid skeleton file: joint assignments are not present\n";
            return;
        }

        // Load all shape keys
        path keyPath = path(model_dir) / "shapekey";
        if (is_directory(keyPath)) {
            int nShapeKeys = 0;
            for (directory_iterator it(keyPath); it != directory_iterator(); ++it) ++nShapeKeys;
            keyClouds.resize(3 * nPoints, nShapeKeys);

            int i = 0;
            for (directory_iterator it(keyPath); it != directory_iterator(); ++it) {
                keyClouds.col(i) = loadPCDToPointVector(it->path().string());
                ++i;
            }
            w.resize(nShapeKeys);
        } else {
            std::cerr << "WARNING: no shape key directory found for avatar\n";
        }

        // Load joint regressor
        std::ifstream jr(jrPath.string());
        jr >> nJoints;
        r.resize(nJoints);

        jointRegressor = Eigen::SparseMatrix<float>(nPoints, nJoints);
        jointRegressor.reserve(nJoints * 10);
        for (int i = 0; i < nJoints; ++i) {
            int nEntries; jr >> nEntries;
            int pointIdx; float val;
            for (int j = 0; j < nEntries; ++j) {
                jr >> pointIdx >> val;
                jointRegressor.insert(pointIdx, i) = val;
            }
            r[i].setIdentity();
        }

        // Process joint assignments
        assignedPoints.resize(nJoints);
        size_t totalAssignments = 0;
        for (int i = 0; i < nPoints; ++i) {
            int nEntries; skel >> nEntries;
            for (int j = 0; j < nEntries; ++j) {
                int joint; float w;
                skel >> joint >> w;
                assignedPoints[joint].emplace_back(i, w);
            }
            totalAssignments += nEntries;
            ii++;
        }

        size_t totalPoints = 0;
        assignStarts.resize(nJoints+1);
        assignVecs.resize(3, totalAssignments);
        assignWeights = Eigen::SparseMatrix<float>(totalAssignments, nPoints);
        assignWeights.reserve(totalAssignments);
        for (int i = 0; i < nJoints; ++i) {
            assignStarts[i] = totalPoints;
            for (auto& assignment : assignedPoints[i]) {
                int p = assignment.first;
                assignWeights.insert(totalPoints, p) = assignment.second;
                assignVecs.col(totalPoints) = baseCloud.segment<3>(p * 3);
                ++totalPoints;
            }
        }
        assignStarts[nJoints] = totalPoints;

        // upate the skin points
        update();
    }

    void HumanAvatar::update() {
        /** Apply shape keys */
        cloud.resize(3, numPoints());
        Eigen::Map<Eigen::VectorXf> cloudVec(cloud.data(), cloud.cols() * 3);
        cloudVec = keyClouds * w + baseCloud; 

        /** Apply joint regressor */
        CloudType jointPosInit = cloud * jointRegressor;

        /** Compute each joint's transform */
        decltype(r) jointRots;
        jointRots.reserve(jointRegressor.cols());
        jointRots.push_back(r[0]);
        jointPos.resize(3, jointPosInit.cols());

        /** Add root position to all joints */
        jointPos.col(0) = p;
        for (size_t i = 1; i < jointRegressor.cols(); ++i) {
            jointRots.push_back(jointRots[parent[i]] * r[i]);
            jointPos.col(i) = jointRots[parent[i]] * (jointPosInit.col(i) -  jointPosInit.col(parent[i])) + jointPos.col(parent[i]);
        }

        /** Compute each point's transform */
        CloudType assignCloud(3, assignWeights.rows());
        for (int i = 0; i < jointRegressor.cols(); ++i) {
            Eigen::Map<CloudType> block(assignCloud.data() + 3 * assignStarts[i], 3, assignStarts[i+1] - assignStarts[i]);
            block = assignVecs.block(0, assignStarts[i], 3, assignStarts[i+1] - assignStarts[i]);
            block.colwise() -= jointPosInit.col(i);
            block = jointRots[i] * block;
            block.colwise() += jointPos.col(i);
        }
        cloud = assignCloud * assignWeights;
    }

    Eigen::VectorXf HumanAvatar::smplParams() const {
        Eigen::VectorXf res;
        res.resize((numJoints() - 1) * 3);
        for (int i = 1; i < numJoints(); ++i) {
            Eigen::AngleAxisf aa;
            aa.fromRotationMatrix(r[i]);
            res = aa.axis() * aa.angle();
        }
        return res;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr HumanAvatar::getCloud() const {
        auto pointCloud = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>();
        pointCloud->resize(numPoints());
        pointCloud->getMatrixXfMap() = cloud;
        return pointCloud;
    }
}
