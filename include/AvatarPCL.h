/**
 * PCL-related utilities for Avatar to get point cloud, get mesh, etc.
 * This is not included in Avatar.h to allow user to opt-out of using PCL
 **/
#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

#include <Avatar.h>

namespace ark {
/** OpenARK Avatar PCL integration */
namespace avatar_pcl {
/** Get PCL point cloud of Avatar's points */
pcl::PointCloud<pcl::PointXYZ>::Ptr getCloud(const Avatar& ava);

/** Get PCL point cloud of Avatar's joints */
pcl::PointCloud<pcl::PointXYZ>::Ptr getJointCloud(const Avatar& ava);

/** Get PCL polygon mesh for avatar's skin
 *  (undefined behavior if avatar has no mesh: !avatar.hasMesh()) */
pcl::PolygonMesh::Ptr getMesh(const Avatar& ava);
}  // namespace avatar_pcl
}  // namespace ark
