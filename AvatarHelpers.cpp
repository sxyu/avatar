#include "internal/AvatarHelpers.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <string>

namespace ark {
/** Hand-written faster function to load a saved PCL point cloud directly
 *  into an Eigen vector, where points are stored: x1 y1 z1 x2 y2 z2 ...
 *  The reason we flatten the cloud instead of using a matrix is to make it
 * easier to add in shape keys, which would otherwise need to be tensors */
Eigen::VectorXd loadPCDToPointVectorFast(const std::string& path) {
    std::ifstream pcd(path);
    int nPoints = -1;
    while (pcd) {
        std::string label;
        pcd >> label;
        if (label == "DATA") {
            if (nPoints < 0) {
                std::cerr << "ERROR: invalid PCD file at " << path
                          << ": no WIDTH field before data, so "
                             "we don't know how many points there are!\n";
                std::exit(0);
            }
            pcd >> label;
            if (label != "ascii") {
                std::cerr << "ERROR: non-ascii PCD not supported! File " << path
                          << "\n";
                std::exit(0);
            }
            break;
        } else if (label == "WIDTH") {
            pcd >> nPoints;
            pcd.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        } else {
            std::string _;
            std::getline(pcd, _);
        }
    }
    if (!pcd || nPoints < 0) {
        std::cerr << "ERROR: invalid PCD file at " << path
                  << ": unexpected EOF\n";
        std::exit(0);
    }

    Eigen::VectorXd result(nPoints * 3);
    for (int i = 0; i < nPoints * 3; ++i) {
        pcd >> result(i);
    }
    return result;
}

/** Spherical to rectangular coords */
void fromSpherical(double rho, double theta, double phi, Eigen::Vector3d& out) {
    out[0] = rho * sin(phi) * cos(theta);
    out[1] = rho * cos(phi);
    out[2] = rho * sin(phi) * sin(theta);
}

template <class T>
void paintTriangleBary(cv::Mat& output, const cv::Size& image_size,
                       const std::vector<cv::Point2f>& projected,
                       const cv::Vec3i& face, const float* zvec, float maxz) {
    std::pair<double, int> yf[3] = {{projected[face(0)].y, 0},
                                    {projected[face(1)].y, 1},
                                    {projected[face(2)].y, 2}};
    std::sort(yf, yf + 3);

    // reorder points for convenience
    auto a = projected[face(yf[0].second)], b = projected[face(yf[1].second)],
         c = projected[face(yf[2].second)];
    a.y = std::floor(a.y);
    c.y = std::ceil(c.y);
    if (a.y == c.y) return;

    int minyi = std::max<int>(a.y, 0),
        maxyi = std::min<int>(c.y, image_size.height - 1),
        midyi = std::floor(b.y);
    float az = zvec[yf[0].second], bz = zvec[yf[1].second],
          cz = zvec[yf[2].second];

    float denom =
        1.0f / ((b.x - c.x) * (a.y - c.y) + (c.y - b.y) * (a.x - c.x));
    if (a.y != b.y) {
        float mhi = (c.x - a.x) / (c.y - a.y);
        float bhi = a.x - a.y * mhi;
        float mlo = (b.x - a.x) / (b.y - a.y);
        float blo = a.x - a.y * mlo;
        if (b.x > c.x) {
            std::swap(mlo, mhi);
            std::swap(blo, bhi);
        }
        for (int i = minyi; i <= std::min(midyi, image_size.height - 1); ++i) {
            int minxi = std::max<int>(std::floor(mlo * i + blo), 0),
                maxxi = std::min<int>(std::ceil(mhi * i + bhi),
                                      image_size.width - 1);
            if (minxi > maxxi) continue;

            float w1v = (b.x - c.x) * (i - c.y);
            float w2v = (c.x - a.x) * (i - c.y);
            T* ptr = output.ptr<T>(i);
            for (int j = minxi; j <= maxxi; ++j) {
                float w1 = (w1v + (c.y - b.y) * (j - c.x)) * denom;
                float w2 = (w2v + (a.y - c.y) * (j - c.x)) * denom;
                ptr[j] = T(std::min(
                    std::max(w1 * az + w2 * bz + (1.f - w1 - w2) * cz, 0.0f),
                    maxz));
            }
        }
    }
    if (b.y != c.y) {
        float mhi = (c.x - a.x) / (c.y - a.y);
        float bhi = a.x - a.y * mhi;
        float mlo = (c.x - b.x) / (c.y - b.y);
        float blo = b.x - b.y * mlo;
        if (b.x > a.x) {
            std::swap(mlo, mhi);
            std::swap(blo, bhi);
        }
        for (int i = std::max(midyi, 0) + (a.y != b.y); i <= maxyi; ++i) {
            int minxi = std::max<int>(std::floor(mlo * i + blo), 0),
                maxxi = std::min<int>(std::ceil(mhi * i + bhi),
                                      image_size.width - 1);
            if (minxi > maxxi) continue;

            float w1v = (b.x - c.x) * (i - c.y);
            float w2v = (c.x - a.x) * (i - c.y);
            T* ptr = output.ptr<T>(i);
            for (int j = minxi; j <= maxxi; ++j) {
                float w1 = (w1v + (c.y - b.y) * (j - c.x)) * denom;
                float w2 = (w2v + (a.y - c.y) * (j - c.x)) * denom;
                ptr[j] = T(std::min(
                    std::max(w1 * az + w2 * bz + (1.f - w1 - w2) * cz, 0.0f),
                    maxz));
            }
        }
    }
}

template void paintTriangleBary<int>(cv::Mat&, const cv::Size&,
                                     const std::vector<cv::Point2f>&,
                                     const cv::Vec3i&, const float*, float);
template void paintTriangleBary<float>(cv::Mat&, const cv::Size&,
                                       const std::vector<cv::Point2f>&,
                                       const cv::Vec3i&, const float*, float);
template void paintTriangleBary<uint8_t>(cv::Mat&, const cv::Size&,
                                         const std::vector<cv::Point2f>&,
                                         const cv::Vec3i&, const float*, float);

/** Paint projected triangle on part mask (CV_8U) using nearest neighbors interp
 */
void paintPartsTriangleNN(
    cv::Mat& output_assigned_joint_mask, const cv::Size& image_size,
    const std::vector<cv::Point2f>& projected,
    const std::vector<std::vector<std::pair<double, int>>>& assigned_joint,
    const cv::Vec3i& face, const std::vector<int>& part_map) {
    std::pair<double, int> xf[3] = {{projected[face[0]].x, 0},
                                    {projected[face[1]].x, 1},
                                    {projected[face[2]].x, 2}};
    std::sort(xf, xf + 3);

    // reorder points for convenience
    auto a = projected[face[xf[0].second]], b = projected[face[xf[1].second]],
         c = projected[face[xf[2].second]];
    a.x = std::floor(a.x);
    c.x = std::ceil(c.x);
    if (a.x == c.x) return;

    auto assigned_a = assigned_joint[face[xf[0].second]][0].second,
         assigned_b = assigned_joint[face[xf[1].second]][0].second,
         assigned_c = assigned_joint[face[xf[2].second]][0].second;
    if (part_map.size()) {
        assigned_a = part_map[assigned_a];
        assigned_b = part_map[assigned_b];
        assigned_c = part_map[assigned_c];
    }

    int minxi = std::max<int>(a.x, 0),
        maxxi = std::min<int>(c.x, image_size.width - 1),
        midxi = std::floor(b.x);

    if (a.x != b.x) {
        double mhi = (c.y - a.y) / (c.x - a.x);
        double bhi = a.y - a.x * mhi;
        double mlo = (b.y - a.y) / (b.x - a.x);
        double blo = a.y - a.x * mlo;
        if (b.y > c.y) {
            std::swap(mlo, mhi);
            std::swap(blo, bhi);
        }
        for (int i = minxi; i <= std::min(midxi, image_size.width - 1); ++i) {
            int minyi = std::max<int>(std::floor(mlo * i + blo), 0),
                maxyi = std::min<int>(std::ceil(mhi * i + bhi),
                                      image_size.height - 1);
            if (minyi > maxyi) continue;

            for (int j = minyi; j <= maxyi; ++j) {
                auto& out = output_assigned_joint_mask.at<uint8_t>(j, i);
                int dista = (a.x - i) * (a.x - i) + (a.y - j) * (a.y - j);
                int distb = (b.x - i) * (b.x - i) + (b.y - j) * (b.y - j);
                int distc = (c.x - i) * (c.x - i) + (c.y - j) * (c.y - j);
                if (dista < distb && dista < distc) {
                    out = assigned_a;
                } else if (distb < distc) {
                    out = assigned_b;
                } else {
                    out = assigned_c;
                }
            }
        }
    }
    if (b.x != c.x) {
        double mhi = (c.y - a.y) / (c.x - a.x);
        double bhi = a.y - a.x * mhi;
        double mlo = (c.y - b.y) / (c.x - b.x);
        double blo = b.y - b.x * mlo;
        if (b.y > a.y) {
            std::swap(mlo, mhi);
            std::swap(blo, bhi);
        }
        for (int i = std::max(midxi, 0) + 1; i <= maxxi; ++i) {
            int minyi = std::max<int>(std::floor(mlo * i + blo), 0),
                maxyi = std::min<int>(std::ceil(mhi * i + bhi),
                                      image_size.height - 1);
            if (minyi > maxyi) continue;

            double w1v = (b.y - c.y) * (i - c.x);
            double w2v = (c.y - a.y) * (i - c.x);
            for (int j = minyi; j <= maxyi; ++j) {
                auto& out = output_assigned_joint_mask.at<uint8_t>(j, i);
                int dista = (a.x - i) * (a.x - i) + (a.y - j) * (a.y - j);
                int distb = (b.x - i) * (b.x - i) + (b.y - j) * (b.y - j);
                int distc = (c.x - i) * (c.x - i) + (c.y - j) * (c.y - j);
                if (dista < distb && dista < distc) {
                    out = assigned_a;
                } else if (distb < distc) {
                    out = assigned_b;
                } else {
                    out = assigned_c;
                }
            }
        }
    }
}

template <class T>
void paintTriangleSingleColor(cv::Mat& output_image, const cv::Size& image_size,
                              const std::vector<cv::Point2f>& projected,
                              const cv::Vec3i& face, T color) {
    std::pair<double, int> yf[3] = {{projected[face[0]].y, 0},
                                    {projected[face[1]].y, 1},
                                    {projected[face[2]].y, 2}};
    std::sort(yf, yf + 3);

    // reorder points for convenience
    auto a = projected[face[yf[0].second]], b = projected[face[yf[1].second]],
         c = projected[face[yf[2].second]];
    a.y = std::floor(a.y);
    c.y = std::ceil(c.y);
    if (a.y == c.y) return;

    int minyi = std::max<int>(a.y, 0),
        maxyi = std::min<int>(c.y, image_size.height - 1),
        midyi = std::floor(b.y);

    if (a.y != b.y) {
        double mhi = (c.x - a.x) / (c.y - a.y);
        double bhi = a.x - a.y * mhi;
        double mlo = (b.x - a.x) / (b.y - a.y);
        double blo = a.x - a.y * mlo;
        if (b.x > c.x) {
            std::swap(mlo, mhi);
            std::swap(blo, bhi);
        }
        for (int i = minyi; i <= std::min(midyi, image_size.height - 1); ++i) {
            int minxi = std::max<int>(std::floor(mlo * i + blo), 0),
                maxxi = std::min<int>(std::ceil(mhi * i + bhi),
                                      image_size.width - 1);
            if (minxi > maxxi) continue;
            T* ptr = output_image.ptr<T>(i);
            std::fill(ptr + minxi, ptr + maxxi, color);
        }
    }
    if (b.y != c.y) {
        double mhi = (c.x - a.x) / (c.y - a.y);
        double bhi = a.x - a.y * mhi;
        double mlo = (c.x - b.x) / (c.y - b.y);
        double blo = b.x - b.y * mlo;
        if (b.x > a.x) {
            std::swap(mlo, mhi);
            std::swap(blo, bhi);
        }
        for (int i = std::max(midyi, 0) + 1; i <= maxyi; ++i) {
            int minxi = std::max<int>(std::floor(mlo * i + blo), 0),
                maxxi = std::min<int>(std::ceil(mhi * i + bhi),
                                      image_size.width - 1);
            if (minxi > maxxi) continue;
            T* ptr = output_image.ptr<T>(i);
            std::fill(ptr + minxi, ptr + maxxi, color);
        }
    }
}

template void paintTriangleSingleColor<int>(cv::Mat&, const cv::Size&,
                                            const std::vector<cv::Point2f>&,
                                            const cv::Vec3i&, int);
template void paintTriangleSingleColor<float>(cv::Mat&, const cv::Size&,
                                              const std::vector<cv::Point2f>&,
                                              const cv::Vec3i&, float);
template void paintTriangleSingleColor<uint8_t>(cv::Mat&, const cv::Size&,
                                                const std::vector<cv::Point2f>&,
                                                const cv::Vec3i&, uint8_t);
}  // namespace ark
