#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "Avatar.h"
#include "AvatarOptimizer.h"
#include "AvatarRenderer.h"
#include "BGSubtractor.h"
#include "Calibration.h"
#include "Config.h"
#include "RTree.h"
#include "Util.h"
#define BEGIN_PROFILE auto start = std::chrono::high_resolution_clock::now()
#define PROFILE(x) do{printf("%s: %f ms\n", #x, std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count()); start = std::chrono::high_resolution_clock::now(); }while(false)

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    std::string datasetPath;
    int bgId, imId, padSize, nnStep;
    float betaPose, betaShape;
    std::string rtreePath;

    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK Basic Background Subtraction Visualization v0.1b (c) Alex Yu 2019\nPosition arguments");
    po::options_description descCombined("");
    desc.add_options()
        ("help", "produce help message")
        ("background,b", po::value<int>(&bgId)->default_value(9999), "Background image id")
        ("image,i", po::value<int>(&imId)->default_value(1), "Current image id")
        ("pad,p", po::value<int>(&padSize)->default_value(4), "Zero pad width for image names in this dataset")
        ("rtree,r", po::value<std::string>(&rtreePath)->default_value(""), "RTree model path")
        ("betapose", po::value<float>(&betaPose)->default_value(0.1), "Optimization loss function: pose prior term weight")
        ("betashape", po::value<float>(&betaShape)->default_value(0.2), "Optimization loss function: shape prior term weight")
        ("nnstep", po::value<int>(&nnStep)->default_value(20), "Optimization nearest-neighbor step: only matches neighbors every x points; a heuristic to improve speed")
    ;
    descPositional.add_options()
        ("dataset_path", po::value<std::string>(&datasetPath)->required(), "Input dataset root directory, should contain depth_exr etc")
    ;
    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
    posopt.add("dataset_path", 1);
    try {
        po::store(po::command_line_parser(argc, argv).options(descCombined)
                .positional(posopt).run(),
                vm);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << descPositional << "\n" << desc << "\n";
        return 1;
    }

    if ( vm.count("help")  )
    {
        std::cout << descPositional << "\n" << desc << "\n";
        return 0;
    }

    try {
        po::notify(vm);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << descPositional << "\n" << desc << "\n";
        return 1;
    }

    using boost::filesystem::path;
    using boost::filesystem::exists;
    std::string intrinPath = (path(datasetPath) / "intrin.txt").string();
    ark::CameraIntrin intrin;
    if (intrinPath.size()) {
        intrin.readFile(intrinPath);
    }

    std::stringstream ss_bg_id;
    ss_bg_id << std::setw(padSize) << std::setfill('0') << std::to_string(bgId);
    std::string bgPath = (path(datasetPath) / "depth_exr" / ("depth_" + ss_bg_id.str() + ".exr")).string();
    cv::Mat background = cv::imread(bgPath, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    if (background.empty()) {
        std::cerr << "ERROR: empty background image. Incorrect path/ID out of bounds/pad size incorrect (specify -p)?\n";
        return 1;
    }
    if (background.channels() == 1) background = intrin.depthToXYZ(background);

    ark::AvatarModel avaModel;
    ark::Avatar ava(avaModel);
    ark::AvatarOptimizer avaOpt(ava, intrin, background.size());
    avaOpt.betaPose = betaPose; 
    avaOpt.betaShape = betaShape;
    avaOpt.nnStep = nnStep;
    ark::BGSubtractor bgsub(background);
    std::vector<std::array<int, 2> > compsBySize;

    ark::RTree rtree(0);
    if (rtreePath.size()) rtree.loadFile(rtreePath);

    bool reinit = true;

    while (true) {
        // std::cerr << imId << " LOAD\n";
        std::stringstream ss_img_id;
        ss_img_id << std::setw(padSize) << std::setfill('0') << std::to_string(imId);

        std::string inPath = (path(datasetPath) / "depth_exr" / ("depth_" + ss_img_id.str() + ".exr")).string();
        cv::Mat image = cv::imread(inPath, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        std::string inPathRGB = (path(datasetPath) / "rgb" / ("rgb_" + ss_img_id.str() + ".jpg")).string();
        cv::Mat imageRGB = cv::imread(inPathRGB);
        if (image.empty() || imageRGB.empty()) {
            std::cerr << "WARNING: no more images found, exiting\n";
            break;
        }
        cv::Mat depth = image;
        if (image.channels() == 1) image = intrin.depthToXYZ(image);
        auto ccstart = std::chrono::high_resolution_clock::now();
        cv::Mat sub = bgsub.run(image, &compsBySize);

        std::vector<int> colorid(256, 255);
        for (int r = 0 ; r < compsBySize.size(); ++r) {
            colorid[compsBySize[r][1]] = r;// > 0 ? 255 : 0;
        }
        cv::Mat vis(sub.size(), CV_8UC3);
        for (int r = 0 ; r < image.rows; ++r) {
            auto* outptr = vis.ptr<cv::Vec3b>(r);
            const auto* inptr = sub.ptr<uint8_t>(r);
            auto* dptr = depth.ptr<float>(r);
            for (int c = 0 ; c < image.cols; ++c) {
                int colorIdx = colorid[inptr[c]];
                if (colorIdx >= 254) {
                    outptr[c] = 0;
                    dptr[c] = 0.0f;
                }
                else outptr[c] = ark::util::paletteColor(colorIdx, true);
            }
        }

        if (rtreePath.size()) {
            cv::Mat result = rtree.predictBest(depth, std::thread::hardware_concurrency());
            for (int r = 0; r < depth.rows; ++r) {
                auto* inPtr = result.ptr<uint8_t>(r);
                auto* visualPtr = vis.ptr<cv::Vec3b>(r);
                for (int c = 0; c < depth.cols; ++c){
                    if (inPtr[c] == 255) continue;
                    visualPtr[c] = ark::util::paletteColor(inPtr[c], true);
                }
            }
            size_t cnz = 0;
            for (int r = 0; r < depth.rows; ++r) {
                auto* partptr = result.ptr<uint8_t>(r);
                for (int c = 0; c < depth.cols; ++c) {
                    if (partptr[c] == 255) continue;
                    ++cnz;
                }
            }
            ark::CloudType dataCloud(3, cnz);
            Eigen::VectorXi dataPartLabels(cnz);
            size_t i = 0;
            for (int r = 0; r < depth.rows; ++r) {
                auto* ptr = image.ptr<cv::Vec3f>(r);
                auto* partptr = result.ptr<uint8_t>(r);
                for (int c = 0; c < depth.cols; ++c) {
                    if (partptr[c] == 255) continue;
                    dataCloud(0, i) = ptr[c][0];
                    dataCloud(1, i) = -ptr[c][1];
                    dataCloud(2, i) = ptr[c][2];
                    dataPartLabels(i) = partptr[c];
                    ++i;
                }
            }
            int icpIters = 3;
            if (reinit) {
                Eigen::Vector3d cloudCen = dataCloud.rowwise().mean();
                ava.p = cloudCen;
                ava.w.setZero();
                for (int i = 1; i < ava.model.numJoints(); ++i) {
                    ava.r[i].setIdentity();
                }
                ava.r[0] = Eigen::AngleAxisd(M_PI, Eigen::Vector3d(0, 1, 0)).toRotationMatrix();
                reinit = false;
                ava.update();
                icpIters = 12;
            }
            BEGIN_PROFILE;
            avaOpt.optimize(dataCloud, dataPartLabels, 16,
                   ark::part_map::SMPL_JOINT_TO_PART_MAP,
                   icpIters,
                   std::thread::hardware_concurrency());
            PROFILE(Optimize (Total));
            ark::AvatarRenderer rend(ava, intrin);
            cv::Mat modelMap = rend.renderDepth(depth.size());
            cv::Vec3b whiteColor(255, 255, 255);
            for (int r = 0 ; r < image.rows; ++r) {
                auto* outptr = vis.ptr<cv::Vec3b>(r);
                const auto* renderptr = modelMap.ptr<float>(r);
                for (int c = 0 ; c < image.cols; ++c) {
                    if (renderptr[c] > 0.0) {
                        outptr[c] = whiteColor * (std::min(1.0, renderptr[c] / 3.0) * 4 / 5) +
                                    outptr[c] / 5;
                    }
                }
            }
            printf("Overall: %f ms\n", std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - ccstart).count());
        }
        for (int r = 0 ; r < image.rows; ++r) {
            auto* outptr = vis.ptr<cv::Vec3b>(r);
            const auto* rgbptr = imageRGB.ptr<cv::Vec3b>(r);
            for (int c = 0 ; c < image.cols; ++c) {
                outptr[c] = rgbptr[c] / 3 + (outptr[c] - rgbptr[c]) / 3 * 2;
            }
        }

        cv::imshow("Visual", vis);
        ++imId;
        int k = cv::waitKey(1);
        if (k == 'q') break;
    }
    return 0;
}

