#include "Avatar.h"
#include <boost/filesystem.hpp>
#include <boost/smart_ptr.hpp>
#include <pcl/io/pcd_io.h>
#include <chrono>

#define BEGIN_PROFILE auto start = std::chrono::high_resolution_clock::now()
#define PROFILE(x) do{printf("%s: %f ms\n", #x, std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count()); start = std::chrono::high_resolution_clock::now(); }while(false)

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
                ++totalPoints;
            }
        }
        assignStarts[nJoints] = totalPoints;

        // upate the skin points
        update();
    }

    void HumanAvatar::update() {
        // BEGIN_PROFILE;

        /** Apply shape keys */
        Eigen::VectorXf shapedCloudVec = keyClouds * w + baseCloud; 
        // PROFILE(ShapeKeys);

        /** Apply joint regressor */
        Eigen::Map<CloudType> shapedCloud(shapedCloudVec.data(), 3, jointRegressor.rows());
        jointPos = shapedCloud * jointRegressor;
        // PROFILE(JointRegr);

        size_t j = 0;
        for (int i = 0; i < jointRegressor.cols(); ++i) {
            auto col = jointPos.col(i);
            for (auto& assignment : assignedPoints[i]) {
                int p = assignment.first;
                assignVecs.col(j++).noalias() = shapedCloud.col(p) - col;
            }
        }

        for (int i = jointRegressor.cols()-1; i >= 0; --i) {
            jointPos.col(i).noalias() -= jointPos.col(parent[i]);
        }

        /** Compute each joint's transform */
        jointRot.clear();
        jointRot.resize(jointRegressor.cols());
        jointRot[0].noalias() = r[0];
        // PROFILE(Alloc2);

        jointPos.col(0) = p; /** Add root position to all joints */
        for (size_t i = 1; i < jointRegressor.cols(); ++i) {
            jointRot[i].noalias() = jointRot[parent[i]] * r[i];
            jointPos.col(i) = jointRot[parent[i]] * jointPos.col(i) + jointPos.col(parent[i]);
        }
        // PROFILE(JointTransform);

        /** Compute each point's transform */
        assignCloud.resize(3, assignWeights.rows());
        for (int i = 0; i < jointRegressor.cols(); ++i) {
            Eigen::Map<CloudType> block(assignCloud.data() + 3 * assignStarts[i], 3, assignStarts[i+1] - assignStarts[i]);
            block.noalias() = assignVecs.block(0, assignStarts[i], 3, assignStarts[i+1] - assignStarts[i]);
            block = jointRot[i] * block;
            block.colwise() += jointPos.col(i);
        }
        // PROFILE(PointTransforms);
        cloud.noalias() = assignCloud * assignWeights;
        // PROFILE(UPDATE New);
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
