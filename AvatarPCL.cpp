#include "Version.h"
#include "Avatar.h"
#include "AvatarPCL.h"

#ifdef OPENARK_PCL_ENABLED
#include <pcl/conversions.h>
#include <boost/smart_ptr.hpp>

namespace ark {
    namespace avatar_pcl {
        pcl::PointCloud<pcl::PointXYZ>::Ptr getCloud(const Avatar& ava) {
            auto pointCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
            pointCloud->points.resize(ava.model.numPoints());
            pointCloud->getMatrixXfMap().block<3, Eigen::Dynamic>(0, 0, 3, pointCloud->size()) = ava.cloud.cast<float>();
            return pointCloud;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr getJointCloud(const Avatar& ava) {
            auto pointCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
            pointCloud->resize(ava.model.numJoints());
            pointCloud->getMatrixXfMap().block<3, Eigen::Dynamic>(0, 0, 3, pointCloud->size()) = ava.jointPos.cast<float>();
            return pointCloud;
        }

        pcl::PolygonMesh::Ptr getMesh(const Avatar& ava) {
            auto pclCloud = getCloud(ava);
            auto mesh = boost::make_shared<pcl::PolygonMesh>();
            pcl::toPCLPointCloud2(*pclCloud, mesh->cloud);

            mesh->polygons.reserve(ava.model.numFaces());
            for (int i = 0; i < ava.model.numFaces();++i) {
                mesh->polygons.emplace_back();
                auto& face = mesh->polygons.back().vertices; 
                face.resize(3);
                for (int j = 0; j < 3; ++j)
                    face[j] = ava.model.mesh(j, i);
            }
            return mesh;
        }
    }
}
#endif
