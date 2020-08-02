#include "Calibration.h"

#include <fstream>

namespace ark {

CameraIntrin::CameraIntrin(const std::string& path) { readFile(path); }

CameraIntrin::CameraIntrin(const cv::Mat& xyz_map) { inferFromXYZ(xyz_map); }

CameraIntrin::CameraIntrin(const cv::Vec4d& intrin) { _setVec4d(intrin); }

void CameraIntrin::clear() {
    fx = fy = cx = cy = 0.f;
    std::fill(k, k + 6, 0.f);
    std::fill(p, p + 2, 0.f);
}

bool CameraIntrin::readFile(const std::string& path) {
    clear();
    std::ifstream ifs(path);
    int good_cnt = 0;
    while (ifs) {
        std::string tag;
        ifs >> tag;
        if (tag.size() != 2) continue;
        if (tag == "cx") {
            ifs >> cx;
            ++good_cnt;
        } else if (tag == "cy") {
            ifs >> cy;
            ++good_cnt;
        } else if (tag == "fx") {
            ifs >> fx;
            ++good_cnt;
        } else if (tag == "fy") {
            ifs >> fy;
            ++good_cnt;
        } else if (tag[0] == 'k') {
            int idx = tag[1] - '1';
            if (idx < 0 || idx >= 6) continue;
            ifs >> k[idx];
        } else if (tag[0] == 'p') {
            int idx = tag[1] - '1';
            if (idx < 0 || idx >= 6) continue;
            ifs >> p[idx];
        }
    }
    // Require cx, cy, fx, fy to exist
    return good_cnt == 4;
}

void CameraIntrin::inferFromXYZ(const cv::Mat& xyz_map) {
    // NOTE: NOT IMPLEMENTED, this is available in the regular OpenARK
    //_setVec4d(util::getCameraIntrinFromXYZ(xyz_map));
    throw "Not implemented";
}

void CameraIntrin::_setVec4d(const cv::Vec4d& intrin) {
    fx = intrin[0];
    cx = intrin[1];
    fy = intrin[2];
    cy = intrin[3];
    std::fill(k, k + 6, 0.f);
    std::fill(p, p + 2, 0.f);
}

Vec3f CameraIntrin::to3D(const Point2f& point, float depth) const {
    Vec3f result;
    result[0] = (point.x - cx) * depth / fx;
    result[1] = (point.y - cy) * depth / fy;
    result[2] = depth;
    return result;
}

Point2f CameraIntrin::to2D(const Vec3f& point) const {
    Point2f result(point[0] * fx / point[2] + cx,
                   point[1] * fy / point[2] + cy);
    return result;
}

cv::Mat CameraIntrin::depthToXYZ(const cv::Mat& depth) const {
    cv::Mat xyz_map(depth.size(), CV_32FC3);
    const float* inPtr;
    cv::Vec3f* outPtr;
    for (int r = 0; r < depth.rows; ++r) {
        inPtr = depth.ptr<float>(r);
        outPtr = xyz_map.ptr<cv::Vec3f>(r);
        for (int c = 0; c < depth.cols; ++c) {
            const float z = inPtr[c];
            outPtr[c] = cv::Vec3f((c - cx) * z / fx, (r - cy) * z / fy, z);
        }
    }
    return xyz_map;
}

bool CameraIntrin::writeFile(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << "fx " << fx << "\ncx " << cx
        << "\n"
           "fy "
        << fy << "\ncy " << cy << "\n";
    for (int i = 0; i < 6; ++i) {
        if (k[i] != 0.f) ofs << "k" << i << " " << k[i] << "\n";
    }
    for (int i = 0; i < 2; ++i) {
        if (p[i] != 0.f) ofs << "p" << i << " " << p[i] << "\n";
    }
    ofs.close();
    return true;
}
}  // namespace ark
