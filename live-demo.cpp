#include <ctime>
#include <cstdlib>
#include <cstdio>
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
#include "Version.h"
// #ifdef PMDSDK_ENABLED
// #include "PMDCamera.h"
// #endif
// #ifdef RSSDK_ENABLED
// #include "SR300Camera.h"
// #endif
// #ifdef RSSDK2_ENABLED
// #include "RS2Camera.h"
// #endif
// #ifdef AZURE_KINECT_ENABLED
#include "AzureKinectCamera.h"
// #endif
#include "BGSubtractor.h"
#include "RTree.h"
#include "Util.h"

#include "opencv2/imgcodecs.hpp"

// #include "Core.h"
// #include "Visualizer.h"
// #include "HumanDetector.h"

using namespace ark;

namespace {
    void readDepth(const std::string & path, cv::Mat & m) {
        std::ifstream ifs(path, std::ios::binary | std::ios::in);

        ushort wid, hi;
        util::read_bin(ifs, hi);
        util::read_bin(ifs, wid);

        m = cv::Mat::zeros(hi, wid, CV_32FC1);

        int zr = 0;
        for (int i = 0; i < hi; ++i) {
            float * ptr = m.ptr<float>(i);
            for (int j = 0; j < wid; ++j) {
                if (zr) --zr;
                else {
                    if (!ifs) break;
                    float x; util::read_bin(ifs, x);
                    if (x <= 1) {
                        ptr[j] = x;
                    }
                    else {
                        zr = (int)(-x) - 1;
                    }
                }
            }
        }
    }
    void writeDepth(const std::string & image_path, cv::Mat & depth_map) {
        std::ofstream ofsd(image_path, std::ios::binary | std::ios::out);

        if (ofsd) {
            util::write_bin(ofsd, (ushort)depth_map.rows);
            util::write_bin(ofsd, (ushort)depth_map.cols);

            int zrun = 0;
            for (int i = 0; i < depth_map.rows; ++i)
            {
                const float * ptr = depth_map.ptr<float>(i);
                for (int j = 0; j < depth_map.cols; ++j)
                {
                    if (ptr[j] == 0) {
                        ++zrun;
                        continue;
                    }
                    else {
                        if (zrun >= 1) {
                            util::write_bin(ofsd, (float)(-zrun));
                        }
                        zrun = 0;
                        util::write_bin(ofsd, ptr[j]);// util::write_bin(ofsd, ptr[j][1]); writeBinary(ofsd, ptr[j][2]);
                    }
                }
            }

            ofsd.close();
        }
    }
}

int main(int argc, char ** argv) {
    namespace po = boost::program_options;
    bool skipRecord;//, jointInference;
    bool forceKinect = false, forceRS2 = false;
    std::string rtreePath;

    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK Avatar Live Demo");
    po::options_description descCombined("");
    desc.add_options()
        ("help", "produce help message")
        ("skip,s", po::bool_switch(&skipRecord), "skip recording (this is quite ointless)")
        // ("infer,i", po::bool_switch(&jointInference), "if set, infers joints using CNN and store joint files")
// #if defined(AZURE_KINECT_ENABLED)
        ("k4a", po::bool_switch(&forceKinect), "if set, prefers Kinect Azure (k4a) depth camera")
// #endif
// #if defined(RSSDK2_ENABLED)
//         ("rs2", po::bool_switch(&forceRS2), "if set, prefers librealsense2 depth cameras")
// #endif
        ("rtree,r", po::value<std::string>(&rtreePath)->default_value(""), "RTree model path")
    ;

    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
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

	printf("CONTROLS:\nQ or ESC to quit\nb to set background\nSPACE to start/pause\n\n");

	// seed the rng
	srand(time(NULL));
    ark::BGSubtractor bgsub{cv::Mat()};

    ark::RTree rtree(0);
    if (rtreePath.size()) rtree.loadFile(rtreePath);

    std::vector<std::array<int, 2> > compsBySize;
    std::vector<int> colorid(256, 255);

    cv::Vec4d intrin;
    if (!skipRecord) {
        // initialize the camera
        DepthCamera::Ptr camera;

// #ifdef AZURE_KINECT_ENABLED
        if (!forceRS2) {
            camera = std::make_shared<AzureKinectCamera>();
        }
// #endif
// #ifdef RSSDK2_ENABLED
//         if (!forceKinect) {
//             camera = std::make_shared<RS2Camera>(true);
//         }
// #endif
// #ifdef RSSDK_ENABLED
//         ASSERT(strcmp(OPENARK_CAMERA_TYPE, "sr300") == 0, "Unsupported RealSense camera type.");
//         camera = std::make_shared<SR300Camera>();
// #endif
// #ifdef PMDSDK_ENABLED
//         camera = std::make_shared<PMDCamera>();
// #endif

// #ifndef AZURE_KINECT_ENABLED
        auto capture_start_time = std::chrono::high_resolution_clock::now();
// #endif

        // turn on the camera
        camera->beginCapture();

        // Read in camera input and save it to the buffer
        std::vector<uint64_t> timestamps;

        // Pausing feature
        bool pause = true;
        std::cerr << "Note: paused, press space to begin recording.\n";

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
                    // cv::rectangle(xyzMap, rect, RECT_COLOR / 255.0, -1);
                    // cv::putText(xyzMap, NO_SIGNAL_STR, STR_POS, 0, 0.8, cv::Scalar(1.0f, 1.0f, 1.0f), 1, cv::LINE_AA);
                }
                else {
                    // store images
                    auto ts = static_cast<AzureKinectCamera*>(camera.get())->getTimestamp();
                    if (timestamps.size() && ts - timestamps.back() < 100000) continue;

                }
                cv::Mat depth;
                cv::extractChannel(xyzMap, depth, 2);
                // visualize
                cv::Mat visual, rgbMapFloat;
                if (!pause) {
                    if (!bgsub.background.empty()) {
                        cv::Mat sub = bgsub.run(xyzMap, &compsBySize);
                        for (int r = 0 ; r < compsBySize.size(); ++r) {
                            colorid[compsBySize[r][1]] = r;// > 0 ? 255 : 0;
                        }
                        for (int r = 0 ; r < xyzMap.rows; ++r) {
                            auto* outptr = rgbMap.ptr<cv::Vec3b>(r);
                            const auto* inptr = sub.ptr<uint8_t>(r);
                            auto* dptr = depth.ptr<float>(r);
                            for (int c = 0 ; c < xyzMap.cols; ++c) {
                                int colorIdx = colorid[inptr[c]];
                                if (colorIdx >= 254) {
                                    dptr[c] = 0;
                                    continue;
                                }
                                if (rtreePath.empty()) {
                                    outptr[c] = outptr[c] / 3.0 + ark::util::paletteColor(colorIdx, true) * 2.0 / 3.0;
                                }
                            }
                        }
                    }

                    if (rtreePath.size()) {
                        cv::Mat result = rtree.predictBest(depth, std::thread::hardware_concurrency());
                        for (int r = 0; r < depth.rows; ++r) {
                            auto* inPtr = result.ptr<uint8_t>(r);
                            auto* visualPtr = rgbMap.ptr<cv::Vec3b>(r);
                            for (int c = 0; c < depth.cols; ++c){
                                if (inPtr[c] == 255) continue;
                                visualPtr[c] = visualPtr[c] / 3.0 + ark::util::paletteColor(inPtr[c], true) * 2.0 / 3.0;
                            }
                        }
                    }
                }

                rgbMap.convertTo(rgbMapFloat, CV_32FC3, 1. / 255.);
                cv::hconcat(xyzMap, rgbMapFloat, visual);
                const int MAX_ROWS = 380;
                if (visual.rows > MAX_ROWS) {
                    cv::resize(visual, visual, cv::Size(MAX_ROWS * visual.cols / visual.rows, MAX_ROWS));
                }
                cv::imshow(camera->getModelName() + " XYZ/RGB Maps", visual);
            }

            int c = cv::waitKey(1);

            // make case insensitive (convert to upper)
            if (c >= 'a' && c <= 'z') c &= 0xdf;
            if (c == 'b') {
                bgsub.background = xyzMap;
                std::cout << "Set background.\n";
            }

            // 27 is ESC
            if (c == 'Q' || c == 27) {
                break;
            }
            else if (c == ' ') {
                if (bgsub.background.empty()) {
                   bgsub.background = xyzMap;
                   std::cout << "Set background.\n";
                }
                pause = !pause;
            }
        }
        camera->endCapture();
        cv::destroyWindow(camera->getModelName() + " XYZ/RGB Maps");
    }

	cv::destroyAllWindows();
	return 0;
}
