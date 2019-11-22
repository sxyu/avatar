#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

// OpenARK Libraries
#include "Avatar.h"
#include "AvatarOptimizer.h"
#include "AvatarRenderer.h"
#include "BGSubtractor.h"
#include "Calibration.h"
#include "Config.h"
#include "RTree.h"
#include "Util.h"

#include "AzureKinectCamera.h"

#include "opencv2/imgcodecs.hpp"

#define BEGIN_PROFILE auto start = std::chrono::high_resolution_clock::now()
#define PROFILE(x) do{printf("%s: %f ms\n", #x, std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count()); start = std::chrono::high_resolution_clock::now(); }while(false)

using namespace ark;

int main(int argc, char ** argv) {
    namespace po = boost::program_options;
    /** Number of body parts from part map, used in rtree, etc. */
    const int numParts = 16;
    std::string intrinPath, rtreePath;
    int nnStep, interval, frameICPIters, reinitICPIters, initialICPIters;
    int initialPerPartCnz, reinitCnz, itersPerICP;
    float betaPose, betaShape;
    bool rtreeOnly, disableOcclusion;
    cv::Size size;

    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK Avatar Live Demo");
    po::options_description descCombined("");
    desc.add_options()
        ("help", "produce help message")
        ("rtree-only,R", po::bool_switch(&rtreeOnly), "Show RTree part segmentation only and skip optimization")
        ("no-occlusion", po::bool_switch(&disableOcclusion), "Disable occlusion detection in avatar optimizer prior to NN matching")
        ("betapose", po::value<float>(&betaPose)->default_value(0.05), "Optimization loss function: pose prior term weight")
        ("betashape", po::value<float>(&betaShape)->default_value(0.12), "Optimization loss function: shape prior term weight")
        ("nnstep", po::value<int>(&nnStep)->default_value(20), "Optimization nearest-neighbor step: only matches neighbors every x points; a heuristic to improve speed (currently, not used)")
        ("data-interval,I", po::value<int>(&interval)->default_value(12), "Only computes rtree weights and optimizes for pixels with x = y = 0 mod interval")
        ("frame-icp-iters,t", po::value<int>(&frameICPIters)->default_value(3), "ICP iterations per frame")
        ("reinit-icp-iters,T", po::value<int>(&reinitICPIters)->default_value(5), "ICP iterations when reinitializing (after tracking loss)")
        ("initial-icp-iters,e", po::value<int>(&initialICPIters)->default_value(7), "ICP iterations when reinitializing (at beginning)")
        ("inner-iters,p", po::value<int>(&itersPerICP)->default_value(10), "Maximum inner iterations per ICP step")
        ("intrin-path,i", po::value<std::string>(&intrinPath)->default_value(""), "Path to camera intrinsics file (default: uses hardcoded K4A intrinsics)")
        ("initial-per-part-thresh", po::value<int>(&initialPerPartCnz)->default_value(80), "Initial detected points per body part (/interval^2) to start tracking avatar")
        ("min-points,M", po::value<int>(&reinitCnz)->default_value(1000), "Minimum number of detected body points to allow continued tracking; if it falls below this number, then the tracker reinitializes")
        ("width", po::value<int>(&size.width)->default_value(1280), "Width of generated images")
        ("height", po::value<int>(&size.height)->default_value(720), "Height of generated imaes")
    ;

    descPositional.add_options()
        ("rtree", po::value<std::string>(&rtreePath), "RTree model path")
    ;

    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
    posopt.add("rtree", 1);
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

	printf("CONTROLS:\nQ or ESC to quit\n"
            "b to set background (also sets when unpausing)\n"
            "SPACE to start/pause\n\n");

	// Seed the rng
    srand(time(NULL));

    CameraIntrin intrin;
    if (!intrinPath.empty()) intrin.readFile(intrinPath);
    else {
        intrin.clear();
        intrin.fx = 606.438;
        intrin.fy = 606.351;
        intrin.cx = 637.294;
        intrin.cy = 366.992;
    }

    ark::AvatarModel avaModel;
    ark::Avatar ava(avaModel);
    ark::AvatarOptimizer avaOpt(ava, intrin, size, numParts,
                                ark::part_map::SMPL_JOINT_TO_PART_MAP);
    avaOpt.betaPose = betaPose;
    avaOpt.betaShape = betaShape;
    avaOpt.nnStep = nnStep;
    avaOpt.enableOcclusion = !disableOcclusion;
    avaOpt.maxItersPerICP = itersPerICP;
    ark::BGSubtractor bgsub{cv::Mat()};
    bgsub.numThreads = std::thread::hardware_concurrency();

    ark::RTree rtree(0);
    if (rtreePath.size()) rtree.loadFile(rtreePath);

    // initialize the camera
    DepthCamera::Ptr camera;

    camera = std::make_shared<AzureKinectCamera>();
    auto capture_start_time = std::chrono::high_resolution_clock::now();

    // turn on the camera
    camera->beginCapture();

    // Read in camera input and save it to the buffer
    std::vector<uint64_t> timestamps;

    // Pausing feature: if true, demo is paused
    bool pause = true;

    // When this flag is true, tracking will reinit the next frame
    bool reinit = true;

    // This indicates if we are reinitializing for the first time
    bool firstTime = true;
    std::cerr << "Note: paused, press space to begin recording.\n"
        "The background (for BG subtraction) will be captured automatically each time you unpause.\nPlease stay out of the grayish area (where depth is unavailable) if possible.\n";

    int currFrame = 0; // current frame number (since launch/last pause)
    while (true)
    {
        ++currFrame;

        // get latest image from the camera
        cv::Mat xyzMap = camera->getXYZMap();
        cv::Mat rgbMap = camera->getRGBMap();

        if (xyzMap.empty() || rgbMap.empty()) {
            std::cerr << "WARNING: Empty image ignored in data recorder loop\n";
        }
        else {
            if (pause) {
                const cv::Scalar RECT_COLOR = cv::Scalar(0, 160, 255);
                const std::string NO_SIGNAL_STR = "PAUSED";
                const cv::Point STR_POS(rgbMap.cols / 2 - 50, rgbMap.rows / 2 + 7);
                const int RECT_WID = 120, RECT_HI = 40;
                cv::Rect rect(rgbMap.cols / 2 - RECT_WID / 2,
                        rgbMap.rows / 2 - RECT_HI / 2,
                        RECT_WID, RECT_HI);

                // show 'paused' and do not record
                cv::rectangle(rgbMap, rect, RECT_COLOR, -1);
                cv::putText(rgbMap, NO_SIGNAL_STR, STR_POS, 0, 0.8, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                // cv::rectangle(xyzMap, rect, RECT_COLOR / 255.0, -1);
                // cv::putText(xyzMap, NO_SIGNAL_STR, STR_POS, 0, 0.8, cv::Scalar(1.0f, 1.0f, 1.0f), 1, cv::LINE_AA);
            }
            else {
                // Capture image if timestamp changed
                auto ts = static_cast<AzureKinectCamera*>(camera.get())->getTimestamp();
                if (timestamps.size() && ts - timestamps.back() < 100000) {
                    continue;
                }

            }

            cv::Mat depth;
            cv::extractChannel(xyzMap, depth, 2);
            // visualize
            cv::Mat visual, rgbMapFloat;
            if (!pause) {
                if (!bgsub.background.empty()) {
                    cv::Mat sub = bgsub.run(xyzMap);
                    size_t subCnz = 0;
                    for (int r = bgsub.topLeft.y ; r <= bgsub.botRight.y; ++r) {
                        // auto* outptr = rgbMap.ptr<cv::Vec3b>(r);
                        const auto* inptr = sub.ptr<uint8_t>(r);
                        auto* dptr = depth.ptr<float>(r);
                        for (int c = bgsub.topLeft.x ; c <= bgsub.botRight.x; ++c) {
                            int colorIdx = inptr[c];
                            if (colorIdx >= 254) {
                                dptr[c] = 0;
                            } else {
                                ++subCnz;
                            }
                        }
                    }

                    if (subCnz < reinitCnz / (interval * interval)) {
                        if (reinit == false) {
                            std::cout << "Note: detected empty scene, decreasing frame rate\n";
                            reinit = true;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    else {
                        cv::Mat result = rtree.predictBest(depth, std::thread::hardware_concurrency(), 2, bgsub.topLeft, bgsub.botRight);
                        rtree.postProcess(result, 2, std::thread::hardware_concurrency(), bgsub.topLeft, bgsub.botRight);
                        if (rtreeOnly) {
                            for (int r = bgsub.topLeft.y; r <= bgsub.botRight.y; ++r) {
                                auto* inPtr = result.ptr<uint8_t>(r);
                                auto* visualPtr = rgbMap.ptr<cv::Vec3b>(r);
                                for (int c = bgsub.topLeft.x; c <= bgsub.botRight.x; ++c){
                                    if (inPtr[c] == 255) continue;
                                    visualPtr[c] = ark::util::paletteColor(inPtr[c], true);
                                }
                            }
                        }
                        else {
                            Eigen::Matrix<size_t, Eigen::Dynamic, 1> partCnz(numParts);
                            partCnz.setZero();
                            for (int r = bgsub.topLeft.y; r <= bgsub.botRight.y; ++r) {
                                auto* partptr = result.ptr<uint8_t>(r);
                                for (int c = bgsub.topLeft.x; c <= bgsub.botRight.x; c += interval) {
                                    if (partptr[c] == 255) continue;
                                    ++partCnz[partptr[c]];
                                }
                            }
                            size_t cnz = partCnz.sum();
                            if ((firstTime && partCnz.minCoeff() < std::max(1, initialPerPartCnz / (interval*interval))) || cnz < reinitCnz / (interval * interval)) {
                                reinit = true;
                            }
                            else {
                                ark::CloudType dataCloud(3, cnz);
                                Eigen::VectorXi dataPartLabels(cnz);
                                size_t i = 0;
                                for (int r = bgsub.topLeft.y ; r <= bgsub.botRight.y; r += interval) {
                                    auto* ptr = xyzMap.ptr<cv::Vec3f>(r);
                                    auto* partptr = result.ptr<uint8_t>(r);
                                    for (int c = bgsub.topLeft.x; c <= bgsub.botRight.x; c += interval) {
                                        if (partptr[c] == 255) continue;
                                        dataCloud(0, i) = ptr[c][0];
                                        dataCloud(1, i) = -ptr[c][1];
                                        dataCloud(2, i) = ptr[c][2];
                                        dataPartLabels(i) = partptr[c];
                                        ++i;
                                    }
                                }
                                int icpIters = frameICPIters;
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
                                    icpIters = firstTime ? initialICPIters : reinitICPIters;
                                    std::cerr << "Note: reinitializing tracking\n";
                                    if (firstTime) firstTime = false;
                                }
                                BEGIN_PROFILE;
                                avaOpt.optimize(dataCloud, dataPartLabels,
                                        icpIters,
                                        std::thread::hardware_concurrency());
                                PROFILE(Optimize (Total));
                                ark::AvatarRenderer rend(ava, intrin);
                                cv::Mat modelMap = rend.renderDepth(depth.size());
                                cv::Vec3b whiteColor(200, 200, 255);
                                for (int r = 0 ; r < rgbMap.rows; ++r) {
                                    auto* outptr = rgbMap.ptr<cv::Vec3b>(r);
                                    const auto* renderptr = modelMap.ptr<float>(r);
                                    for (int c = 0 ; c < rgbMap.cols; ++c) {
                                        if (renderptr[c] > 0.0) {
                                            outptr[c] = whiteColor *
                                                (std::min(1.0, renderptr[c] / 3.0) * 4 / 5) +
                                                outptr[c] / 5;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            rgbMap.convertTo(rgbMapFloat, CV_32FC3, 1. / 255.);
            // cv::hconcat(xyzMap, rgbMapFloat, visual);
            visual = rgbMapFloat;
            for (int r = 0 ; r < visual.rows; ++r) {
                auto* outptr = visual.ptr<cv::Vec3f>(r);
                auto* xyzptr = xyzMap.ptr<cv::Vec3f>(r);
                for (int c = 0 ; c < visual.cols; ++c) {
                    if (xyzptr[c][2] == 0.0f) {
                        // Since depth camera's FoV is actually
                        // smaller than RGB FoV visible,
                        // try to make user aware of
                        // depth boundaries by darkening areas with
                        // no depth info
                        outptr[c] *= 0.5;
                    }
                }
            }
            const int MAX_COLS = 1300;
            if (visual.cols > MAX_COLS) {
                cv::resize(visual, visual, cv::Size(MAX_COLS, MAX_COLS * visual.rows / visual.cols));
            }
            cv::imshow(camera->getModelName() + " XYZ/RGB Maps", visual);
        }

        int c = cv::waitKey(1);

        // make case insensitive (convert to upper)
        if (c >= 'a' && c <= 'z') c &= 0xdf;
        if (c == 'b') {
            bgsub.background = xyzMap;
            std::cout << "Note: background updated.\n";
        }

        // 27 is ESC
        if (c == 'Q' || c == 27) {
            break;
        }
        else if (c == ' ') {
            if (bgsub.background.empty()) {
                bgsub.background = xyzMap;
                std::cout << "Note: background updated.\n";
            }
            pause = !pause;
            if (pause) reinit = true;
        }
    }
    camera->endCapture();
    cv::destroyWindow(camera->getModelName() + " XYZ/RGB Maps");

    cv::destroyAllWindows();
    return 0;
}
