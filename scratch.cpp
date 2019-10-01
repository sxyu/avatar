#include <vector>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/visualization/pcl_visualizer.h>

#include "Avatar.h"
#include "AvatarPCL.h"
#include "AvatarRenderer.h"
#include "ViconSkeleton.h"
#include "Util.h"
#define BEGIN_PROFILE auto start = std::chrono::high_resolution_clock::now()
#define PROFILE(x) do{printf("%s: %f ms\n", #x, std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count()); start = std::chrono::high_resolution_clock::now(); }while(false)

constexpr char WIND_NAME[] = "Result";

// open a gui for interacting with avatar
void __avatarGUI()
{
    using namespace ark;
    // build file names and paths
    AvatarModel model;
    Avatar ava(model);

    const size_t NKEYS = model.numShapeKeys();

    cv::namedWindow("Body Shape");
    cv::namedWindow("Body Pose");
    std::vector<int> pcw(NKEYS, 1000), p_pcw(NKEYS, 0);

    // define some axes
    const Eigen::Vector3d AXISX(1, 0, 0), AXISY(0, 1, 0), AXISZ(0, 0, 1);

    // Body pose control definitions (currently this control system only supports rotation along one axis per body part)
    const std::vector<std::string> CTRL_NAMES       = {"L HIP",      "R HIP",      "L KNEE",      "R KNEE",      "L ANKLE",      "R ANKLE",      "L ELBLW",        "R ELBOW",        "L WRIST",      "R WRIST",      "HEAD",      "SPINE2",     "ROOT"};
    const std::vector<int> CTRL_JNT               = {SmplJoint::L_HIP, SmplJoint::R_HIP, SmplJoint::L_KNEE, SmplJoint::R_KNEE, SmplJoint::L_ANKLE, SmplJoint::R_ANKLE, SmplJoint::L_ELBOW, SmplJoint::R_ELBOW, SmplJoint::L_WRIST, SmplJoint::R_WRIST, SmplJoint::HEAD, SmplJoint::SPINE2, SmplJoint::ROOT_PELVIS};
    const std::vector<Eigen::Vector3d> CTRL_AXIS    = {AXISX,        AXISX,        AXISX,         AXISX,         AXISX,          AXISX,          AXISY,          AXISY,          AXISY,          AXISY,          AXISX,       AXISX,         AXISY};
    const int N_CTRL = (int)CTRL_NAMES.size();

    std::vector<int> ctrlw(N_CTRL, 1000), p_ctrlw(N_CTRL, 0);

    // Body shapekeys are defined in SMPL model files.
    int pifx = 0, pify = 0, picx = 0, picy = 0, pframeID = -1;
    cv::resizeWindow("Body Shape", cv::Size(400, 700));
    cv::resizeWindow("Body Pose", cv::Size(400, 700));
    cv::resizeWindow("Body Scale", cv::Size(400, 700));
    for (int i = 0; i < N_CTRL; ++i) {
        cv::createTrackbar(CTRL_NAMES[i], "Body Pose", &ctrlw[i], 2000);
    }
    for (int i = 0; i < (int)pcw.size(); ++i) {
        cv::createTrackbar("PC" + std::to_string(i), "Body Shape", &pcw[i], 2000);
    }

    auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewport"));

    viewer->initCameraParameters();
    int vp1 = 0;
    viewer->setWindowName("3D View");
    viewer->setBackgroundColor(0, 0, 0);
    // viewer->setCameraClipDistances(0.0, 1000.0);

    volatile bool interrupt = false;
    viewer->registerKeyboardCallback([&interrupt](const pcl::visualization::KeyboardEvent & evt) {
        unsigned char k = evt.getKeyCode();
        if (k == 'Q' || k == 'q' || k == 27) {
            interrupt = true;
        }
    });

    while (!interrupt) {
        bool controlsChanged = false;
        for (int i = 0; i < N_CTRL; ++i) {
            if (ctrlw[i] != p_ctrlw[i]) {
                controlsChanged = true;
                break;
            }
        }
        for (int i = 0; i < (int)pcw.size(); ++i) {
            if (pcw[i] != p_pcw[i]) {
                controlsChanged = true;
                break;
            }
        }
        if (controlsChanged) {
            ava.update();

            viewer->removePointCloud("vp1_cloudHM");
            viewer->addPointCloud<pcl::PointXYZ>(avatar_pcl::getCloud(ava), "vp1_cloudHM", vp1);
            viewer->removePolygonMesh("meshHM");

            auto mesh = ark::avatar_pcl::getMesh(ava);
            viewer->addPolygonMesh(*mesh, "meshHM", vp1);
            //ava.visualize(viewer, "vp1_", vp1);

            for (int i = 0; i < N_CTRL; ++i) {
                double angle = (ctrlw[i] - 1000) / 1000.0 * M_PI;
                if (CTRL_JNT[i] >= ava.model.numJoints()) continue;
                if (angle == 0) ava.r[CTRL_JNT[i]].setIdentity();
                else ava.r[CTRL_JNT[i]] = Eigen::AngleAxisd(angle, CTRL_AXIS[i]).toRotationMatrix();
            }

            for (int i = 0; i < (int)pcw.size(); ++i) {
                ava.w[i] = (float)(pcw[i] - 1000) / 500.0;
            }

            ava.p = Eigen::Vector3d(0, 0, 0);
            ava.update();

            for (int k = 0; k < (int) pcw.size(); ++k) {
                p_pcw[k] = pcw[k] = (int) (ava.w[k] * 500.0 + 1000);
                cv::setTrackbarPos("PC" + std::to_string(k), "Body Shape", pcw[k]);
            }

            double prior = ava.model.posePrior.residual(ava.smplParams()).squaredNorm();
            // show pose prior value
            if (!viewer->updateText("-log likelihood: " + std::to_string(prior), 10, 20, 15, 1.0, 1.0, 1.0, "poseprior_disp")) {
                viewer->addText("-log likelihood: " + std::to_string(prior), 10, 20, 15, 1.0, 1.0, 1.0, "poseprior_disp");
            }

        }
        for (int i = 0; i < N_CTRL; ++i) p_ctrlw[i] = ctrlw[i];
        for (int i = 0; i < (int)pcw.size(); ++i) p_pcw[i] = pcw[i];

        int k = cv::waitKey(1);
        viewer->spinOnce();
        if (k == 'q' || k == 27) break;
    }
}

// void avatarViconAlign()
// {
//     using namespace ark;
//     // build file names and paths
//     AvatarModel model;
//     Avatar ava(model);
//
//     const size_t NKEYS = model.numShapeKeys();
//
//     cv::namedWindow("Body Shape");
//     std::vector<int> pcw(NKEYS, 1000), p_pcw(NKEYS, 0);
//
//     // define some axes
//     const Eigen::Vector3d AXISX(1, 0, 0), AXISY(0, 1, 0), AXISZ(0, 0, 1);
//
//     // Body pose control definitions (currently this control system only supports rotation along one axis per body part)
//
//     // Body shapekeys are defined in SMPL model files.
//     int pifx = 0, pify = 0, picx = 0, picy = 0, pframeID = -1;
//     cv::resizeWindow("Body Shape", cv::Size(400, 700));
//     for (int i = 1; i < (int)pcw.size(); ++i) {
//         cv::createTrackbar("PC" + std::to_string(i), "Body Shape", &pcw[i], 2000);
//     }
//
//     auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewport"));
//
//     viewer->initCameraParameters();
//     int vp1 = 0;
//     viewer->setWindowName("3D View");
//     viewer->setBackgroundColor(0, 0, 0);
//     // viewer->setCameraClipDistances(0.0, 1000.0);
//
//     volatile bool interrupt = false;
//     viewer->registerKeyboardCallback([&interrupt](const pcl::visualization::KeyboardEvent & evt) {
//         unsigned char k = evt.getKeyCode();
//         if (k == 'Q' || k == 'q' || k == 27) {
//             interrupt = true;
//         }
//     });
//
//     CameraIntrin intrin;
//     intrin.clear();
//     intrin.fx = 606.438;
//     intrin.fy = 606.351;
//     intrin.cx = 637.294;
//     intrin.cy = 366.992;
//
//     cv::Size size(1280, 720);
//
//     std::cerr << "BEFORE\n";
//     ViconSkeleton skel("/mnt_d/DataSets/human/CMUMocap/mocapPlayer/07.asf", "/mnt_d/DataSets/human/CMUMocap/mocapPlayer/07_05-walk.amc", 1);
//     std::cerr << "LOAD\n";
//
//     bool controlsChanged = false;
//     while (!interrupt) {
//         if (!controlsChanged) {
//             for (int i = 1; i < (int)pcw.size(); ++i) {
//                 if (pcw[i] != p_pcw[i]) {
//                     controlsChanged = true;
//                     break;
//                 }
//             }
//         }
//
//         if (controlsChanged) {
//             skel.nextFrame();
//             std::cerr << skel.frame() << " FRAME\n";
//             CloudType smplJoints = skel.getSmplJoints();
//             for (int i = 0; i < smplJoints.cols(); ++i) {
//                 pcl::PointXYZRGBA curr;
//                 curr.getVector3fMap() = smplJoints.col(i).cast<float>();
//                 std::string jointName = "r_joint" + std::to_string(i);
//                 viewer->removeShape(jointName);
//                 viewer->addSphere(curr, 0.02, 0.1, 1.0, 0.0, jointName, 0);
//
//                 viewer->removeText3D(jointName + "T");
//                 viewer->addText3D(std::to_string(i), curr, 0.03, 1.0, 1.0, 1.0, jointName + "T", 0);
//             }
//             ava.alignToJoints(smplJoints);
//             //viewer->removeAllPointClouds(vp1);
//             //viewer->removeAllShapes(vp1);
//             // ava.update();
//
//             //viewer->removePointCloud("vp1_cloudHM");
//             //viewer->addPointCloud<pcl::PointXYZ>(avatar_pcl::getCloud(ava), "vp1_cloudHM", vp1);
//             // viewer->removePolygonMesh("meshHM");
//
//             // auto mesh = ark::avatar_pcl::getMesh(ava);
//             // viewer->addPolygonMesh(*mesh, "meshHM", vp1);
//             //ava.visualize(viewer, "vp1_", vp1);
//
//             // for (int i = 0; i < N_CTRL; ++i) {
//             //     double angle = (ctrlw[i] - 1000) / 1000.0 * M_PI;
//             //     if (angle == 0) ava.r[CTRL_JNT[i]].setIdentity();
//             //     else ava.r[CTRL_JNT[i]] = Eigen::AngleAxisd(angle, CTRL_AXIS[i]).toRotationMatrix();
//             // }
//
//             for (int i = 1; i < (int)pcw.size(); ++i) {
//                 ava.w[i] = (float)(pcw[i] - 1000) / 500.0;
//             }
//
//             ava.update();
//
//             for (int k = 1; k < (int) pcw.size(); ++k) {
//                 p_pcw[k] = pcw[k] = (int) (ava.w[k] * 500.0 + 1000);
//                 cv::setTrackbarPos("PC" + std::to_string(k), "Body Shape", pcw[k]);
//             }
//
//             // renderer.update();
//             // cv::Mat facesMap = renderer.renderFaces(size);
//             // const auto& faces = renderer.getOrderedFaces();
//             // std::vector<int> faceIndices;
//             // faceIndices.reserve(facesMap.rows * facesMap.cols / 10);
//             // for (int r = 0; r < facesMap.rows; ++r) {
//             //     auto* ptr = facesMap.ptr<int32_t>(r);
//             //     for (int c = 0; c < facesMap.cols; ++c) {
//             //         if (~ptr[c]) {
//             //             faceIndices.push_back(ptr[c]);
//             //         }
//             //     }
//             // }
//             // // std::sort(faceIndices.begin(), faceIndices.end());
//             // // faceIndices.resize(std::unique(faceIndices.begin(), faceIndices.end()) - faceIndices.begin());
//             // std::vector<int> pointVisible(ava.cloud.size());
//             // for (int i : faceIndices) {
//             //     pointVisible[faces[i].second[0]]
//             //         = pointVisible[faces[i].second[1]]
//             //         = pointVisible[faces[i].second[2]] = true;
//             // }
//
//             auto visualCloud = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA> >(new pcl::PointCloud<pcl::PointXYZRGBA>());
//             for (size_t i = 0; i < ava.cloud.cols(); ++i) {
//                 // if (pointVisible[i]) {
//                     pcl::PointXYZRGBA pt;
//                     pt.x = ava.cloud(0, i);
//                     pt.y = ava.cloud(1, i);
//                     pt.z = ava.cloud(2, i);
//                     pt.r = 255;
//                     pt.g = 0;
//                     pt.b = 255;
//                     pt.a = 255;
//                     visualCloud->push_back(std::move(pt));
//                 // }
//             }
//
//             viewer->removePointCloud("vp1_cloudHMC");
//             viewer->addPointCloud<pcl::PointXYZRGBA>(visualCloud, "vp1_cloudHMC");
//             // ava.visualize(viewer, "vp1_", vp1);
//             //
//             // cv::Mat visual(facesMap.size(), CV_8UC3);
//             // for (int r = 0; r < facesMap.rows; ++r) {
//             //     auto ptr = facesMap.ptr<int32_t>(r);
//             //     auto outPtr = visual.ptr<cv::Vec3b>(r);
//             //     for (int c = 0; c < facesMap.cols; ++c) {
//             //         outPtr[c] = ark::util::paletteColor(ptr[c], true);
//             //     }
//             // }
//             // cv::imshow(WIND_NAME, visual);
//         }
//         for (int i = 0; i < (int)pcw.size(); ++i) p_pcw[i] = pcw[i];
//
//         int k = cv::waitKey(1);
//         viewer->spinOnce();
//         controlsChanged = false;
//         if (k == 'q' || k == 27) break;
//         else if (k == 'n') controlsChanged = true;
//     }
// }

int main(int argc, char** argv) {
    __avatarGUI();
    // avatarViconAlign();
    return 0;
}
