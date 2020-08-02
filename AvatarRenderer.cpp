#include "AvatarRenderer.h"

#include <iostream>
#include "internal/AvatarHelpers.h"

namespace ark {

AvatarRenderer::AvatarRenderer(const Avatar& ava, const CameraIntrin& intrin)
    : ava(ava), intrin(intrin) {}

const std::vector<cv::Point2f>& AvatarRenderer::getProjectedPoints() const {
    if (projectedPoints.empty()) {
        projectedPoints.resize(ava.model.numPoints());
        for (size_t i = 0; i < ava.cloud.cols(); ++i) {
            const auto& pt = ava.cloud.col(i);
            projectedPoints[i].x =
                static_cast<double>(pt(0)) * intrin.fx / pt(2) + intrin.cx;
            projectedPoints[i].y =
                -static_cast<double>(pt(1)) * intrin.fy / pt(2) + intrin.cy;
        }
    }
    return projectedPoints;
}

const std::vector<cv::Point2f>& AvatarRenderer::getProjectedJoints() const {
    if (projectedJoints.empty()) {
        projectedJoints.resize(ava.model.numJoints());
        for (size_t i = 0; i < ava.jointPos.cols(); ++i) {
            const auto& pt = ava.jointPos.col(i);
            projectedJoints[i].x =
                static_cast<double>(pt(0)) * intrin.fx / pt(2) + intrin.cx;
            projectedJoints[i].y =
                -static_cast<double>(pt(1)) * intrin.fy / pt(2) + intrin.cy;
        }
    }
    return projectedJoints;
}

const std::vector<AvatarRenderer::FaceType>& AvatarRenderer::getOrderedFaces()
    const {
    if (orderedFaces.empty()) {
        static auto faceComp = [](const FaceType& a, const FaceType& b) {
            return a.first > b.first;
        };

        orderedFaces.reserve(ava.model.numFaces());
        for (int i = 0; i < ava.model.numFaces(); ++i) {
            const auto& face = ava.model.mesh.col(i);
            orderedFaces.emplace_back(0.f,
                                      cv::Vec3i(face(0), face(1), face(2)));
        }

        if (ava.cloud.cols() == 0) {
            std::cerr << "WARNING: Attempt to render empty avatar detected, "
                         "please call update() first\n";
            return orderedFaces;
        }
        // Sort faces by decreasing center depth
        // so that when painted front faces will cover back faces
        for (int i = 0; i < ava.model.numFaces(); ++i) {
            auto& face = orderedFaces[i].second;
            orderedFaces[i].first =
                (ava.cloud(2, face[0]) + ava.cloud(2, face[1]) +
                 ava.cloud(2, face[2])) /
                3.f;
        }
        std::sort(orderedFaces.begin(), orderedFaces.end(), faceComp);
    }
    return orderedFaces;
}

cv::Mat AvatarRenderer::renderDepth(const cv::Size& image_size) const {
    if (ava.cloud.cols() == 0) {
        std::cerr << "WARNING: Attempt to render empty avatar detected, please "
                     "call update() first\n";
        return cv::Mat();
    }
    const auto& projected = getProjectedPoints();
    const auto& faces = getOrderedFaces();

    cv::Mat renderedDepth = cv::Mat::zeros(image_size, CV_32F);
    float zv[3];
    for (int i = 0; i < ava.model.numFaces(); ++i) {
        auto& a = ava.cloud.col(faces[i].second[0]);
        auto& b = ava.cloud.col(faces[i].second[1]);
        auto& c = ava.cloud.col(faces[i].second[2]);
        Eigen::Vector3d ab = b - a, ac = c - a;
        double zcross = fabs(ab.cross(ac).normalized().z());
        if (zcross < 0.1) {
            paintTriangleSingleColor(renderedDepth, image_size, projected,
                                     faces[i].second, 0.f);
        } else {
            zv[0] = (float)a.z();
            zv[1] = (float)b.z();
            zv[2] = (float)c.z();
            paintTriangleBary<float>(renderedDepth, image_size, projected,
                                     faces[i].second, zv);
        }
    }
    return renderedDepth;
}

cv::Mat AvatarRenderer::renderLambert(const cv::Size& image_size) const {
    if (ava.cloud.cols() == 0) {
        std::cerr << "WARNING: Attempt to render empty avatar detected, please "
                     "call update() first\n";
        return cv::Mat();
    }
    const auto& projected = getProjectedPoints();
    const auto& faces = getOrderedFaces();

    cv::Mat renderedGray = cv::Mat::zeros(image_size, CV_8U);
    const Eigen::Vector3d mainLight(0.8, 1.5, -1.2);
    const double mainLightIntensity = 0.8;
    const Eigen::Vector3d backLight(-0.2, -1.5, 0.4);
    const double backLightIntensity = 0.2;
    float lambert[3];

    std::vector<bool> visible(ava.model.numFaces());
    Eigen::Matrix<double, 3, Eigen::Dynamic> vertNormal(3,
                                                        ava.model.numPoints());
    vertNormal.setZero();
    for (int i = 0; i < ava.model.numFaces(); ++i) {
        auto& a = ava.cloud.col(faces[i].second[0]);
        auto& b = ava.cloud.col(faces[i].second[1]);
        auto& c = ava.cloud.col(faces[i].second[2]);
        Eigen::Vector3d normal = (b - a).cross(c - a).normalized();
        for (int j = 0; j < 3; ++j) {
            vertNormal.col(faces[i].second[j]) += normal;
        }
        visible[i] = abs(normal.z()) > 1e-2;
    }
    vertNormal.colwise().normalize();
    for (int i = 0; i < ava.model.numPoints(); ++i) {
        auto normal = vertNormal.col(i);
        if (normal.z() > 0) normal = -normal;
    }

    for (int i = 0; i < ava.model.numFaces(); ++i) {
        if (!visible[i]) continue;
        int ai = faces[i].second[0], bi = faces[i].second[1],
            ci = faces[i].second[2];
        auto a = ava.cloud.col(ai), b = ava.cloud.col(bi),
             c = ava.cloud.col(ci);
        auto na = vertNormal.col(ai), nb = vertNormal.col(bi),
             nc = vertNormal.col(ci);
        Eigen::Vector3d mainLightVec_a = (mainLight - a).normalized();
        Eigen::Vector3d backLightVec_a = (backLight - a).normalized();
        lambert[0] =
            std::max(float(mainLightVec_a.dot(na) * mainLightIntensity +
                           backLightVec_a.dot(na) * backLightIntensity) *
                         255,
                     0.f);
        Eigen::Vector3d mainLightVec_b = (mainLight - b).normalized();
        Eigen::Vector3d backLightVec_b = (backLight - b).normalized();
        lambert[1] =
            std::max(float(mainLightVec_b.dot(nb) * mainLightIntensity +
                           backLightVec_b.dot(nb) * backLightIntensity) *
                         255,
                     0.f);
        Eigen::Vector3d mainLightVec_c = (mainLight - c).normalized();
        Eigen::Vector3d backLightVec_c = (backLight - c).normalized();
        lambert[2] =
            std::max(float(mainLightVec_c.dot(nc) * mainLightIntensity +
                           backLightVec_c.dot(nc) * backLightIntensity) *
                         255,
                     0.f);
        paintTriangleBary<uint8_t>(renderedGray, image_size, projected,
                                   faces[i].second, lambert);
    }
    return renderedGray;
}

cv::Mat AvatarRenderer::renderPartMask(const cv::Size& image_size,
                                       const std::vector<int>& part_map) const {
    if (ava.cloud.cols() == 0) {
        std::cerr << "WARNING: Attempt to render empty avatar detected\n";
        return cv::Mat();
    }
    const auto& projected = getProjectedPoints();
    const auto& faces = getOrderedFaces();

    cv::Mat partMaskMap = cv::Mat::zeros(image_size, CV_8U);
    partMaskMap.setTo(255);
    for (int i = 0; i < ava.model.numFaces(); ++i) {
        auto& a = ava.cloud.col(faces[i].second[0]);
        auto& b = ava.cloud.col(faces[i].second[1]);
        auto& c = ava.cloud.col(faces[i].second[2]);
        Eigen::Vector3d ab = b - a, ac = c - a;
        double zcross = fabs(ab.cross(ac).normalized().z());
        if (zcross < 0.1) {
            paintTriangleSingleColor<uint8_t>(partMaskMap, image_size,
                                              projected, faces[i].second,
                                              uint8_t(255));
        } else {
            paintPartsTriangleNN(partMaskMap, image_size, projected,
                                 ava.model.assignedJoints, faces[i].second,
                                 part_map);
        }
    }
    return partMaskMap;
}

cv::Mat AvatarRenderer::renderFaces(const cv::Size& image_size,
                                    int num_threads) const {
    const auto& projected = getProjectedPoints();
    const auto& faces = getOrderedFaces();

    cv::Mat facesMap = cv::Mat::zeros(image_size, CV_32S);
    facesMap.setTo(-1);
    for (int i = 0; i < ava.model.numFaces(); ++i) {
        paintTriangleSingleColor(facesMap, image_size, projected,
                                 faces[i].second, i);
    }
    return facesMap;
}

void AvatarRenderer::update() const {
    projectedPoints.clear();
    projectedJoints.clear();
    orderedFaces.clear();
}

}  // namespace ark
