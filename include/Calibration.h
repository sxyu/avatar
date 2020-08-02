#pragma once

#include "Version.h"
#include <string>
#include <opencv2/core.hpp>

namespace ark {
/**
 * Single camera intrinsic data
 */
struct CameraIntrin {
    // Focal length, principal point
    float fx, fy, cx, cy;
    // Radial distortion
    float k[6];
    // Tangential distortion
    float p[2];

    /** Default constructor */
    CameraIntrin() {}

    /** Load from path */
    explicit CameraIntrin(const std::string& path);

    /** Infer from XYZ map (CV_32FC3) */
    explicit CameraIntrin(const cv::Mat& xyz_map);

    /** Convert from legacy format, for internal use */
    explicit CameraIntrin(const cv::Vec4d& intrin);

    /** Clear parameters */
    void clear();

    template <class T>
    /** Copy from other intrinsics struct e.g. from camera vendor, which should
     * have fx, fy, cx, cy, etc. */
    void copyFrom(const T& intrin) {
        fx = intrin.fx;
        fy = intrin.fy;
        cx = intrin.cx;
        cy = intrin.cy;
        k[0] = intrin.k1;
        k[1] = intrin.k2;
        k[2] = intrin.k3;
        k[3] = intrin.k4;
        k[4] = intrin.k5;
        k[5] = intrin.k6;
        p[0] = intrin.p1;
        p[1] = intrin.p2;
    }

    /** Infer from XYZ map (CV_32FC3) */
    void inferFromXYZ(const cv::Mat& xyz_map);

    /** Convert from legacy format, for internal use */
    void _setVec4d(const cv::Vec4d& intrin);

    /** Convert a 2D screen point to 3D point in camera space */
    Vec3f to3D(const Point2f& point, float depth) const;

    /** Project a 3D point in camera space to 2D screen space */
    Point2f to2D(const Vec3f& point) const;

    /** Convert depth map to XYZ
     *  @param depth depth map, CV_32FC1 format
     *  @return XYZ map, CV_32FC3 format
     **/
    cv::Mat depthToXYZ(const cv::Mat& depth) const;

    /** Write to file
     * @return true on success */
    bool writeFile(const std::string& path) const;

    /** Load from path
     * @return true on success */
    bool readFile(const std::string& path);
};
}  // namespace ark
