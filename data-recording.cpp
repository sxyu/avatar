
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cmath>
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

#include "Version.h"

#ifdef OPENARK_AZURE_KINECT_ENABLED
#include "AzureKinectCamera.h"
#endif
#ifdef OPENARK_FREENECT2_ENABLED
#include "Freenect2Camera.h"
#endif

#include "Util.h"

#include "opencv2/imgcodecs.hpp"

// #include "Core.h"
// #include "Visualizer.h"
// #include "HumanDetector.h"

using namespace ark;

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    std::string outPath;
    bool skipRecord, verify;
    bool forceK4a = false, forceFreenect2 = false;

    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK Data Recording Tool");
    po::options_description descCombined("");
    desc.add_options()("help", "produce help message")(
        "skip,s", po::bool_switch(&skipRecord),
        "skip recording (this is quite pointless)")(
        "verify", po::bool_switch(&verify), "verify data by loading from disk")
#ifdef OPENARK_AZURE_KINECT_ENABLED
        ("k4a", po::bool_switch(&forceK4a),
         "if set, forces Kinect Azure (k4a) depth camera")
#endif
#ifdef OPENARK_FREENECT2_ENABLED
            ("freenect2", po::bool_switch(&forceFreenect2),
             "if set, forces Freenect2 depth camera")
#endif
        // #if defined(OPENARK2_RSSDK2_ENABLED)
        //         ("rs2", po::bool_switch(&forceRS2), "if set, prefers
        //         librealsense2 depth cameras")
        // #endif
        ;

    descPositional.add_options()("output_path",
                                 po::value<std::string>(&outPath)->required(),
                                 "Output Path");
    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
    posopt.add("output_path", 1);
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(descCombined)
                      .positional(posopt)
                      .run(),
                  vm);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << descPositional << "\n" << desc << "\n";
        return 1;
    }

    if (vm.count("help")) {
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

    if ((int)forceK4a + (int)forceFreenect2 > 1) {
        std::cerr << "Only one camera preference may be provided";
        return 1;
    }

    printf(
        "CONTROLS:\nQ or ESC to stop recording,\nSPACE to start/pause"
        "(warning: if pausing in the middle, may mess up timestamps)\n\n");

    // seed the rng
    srand(time(NULL));

    using boost::filesystem::path;
    const path directory_path(outPath);

    path depth_path = directory_path / "depth_exr/";
    path rgb_path = directory_path / "rgb/";
    path timestamp_path = directory_path / "timestamp.txt";
    path intrin_path = directory_path / "intrin.txt";
    if (!boost::filesystem::exists(depth_path)) {
        boost::filesystem::create_directories(depth_path);
    }
    if (!boost::filesystem::exists(rgb_path)) {
        boost::filesystem::create_directories(rgb_path);
    }
    cv::Mat lastXYZMap;
    if (!skipRecord) {
        // initialize the camera
        DepthCamera::Ptr camera;

        if (forceK4a) {
#ifdef OPENARK_AZURE_KINECT_ENABLED
            camera = std::make_shared<AzureKinectCamera>();
#endif
        } else if (forceFreenect2) {
#ifdef OPENARK_FREENECT2_ENABLED
            camera = std::make_shared<Freenect2Camera>();
#endif
        } else {
            camera = std::make_shared<OPENARK_PREFERRED_CAMERA>();
        }

        std::cerr << "Starting data recording, saving to: "
                  << directory_path.string() << "\n";
        auto capture_start_time = std::chrono::high_resolution_clock::now();

        // turn on the camera
        camera->beginCapture();

        // If failed to opened camera
        if (!camera->isCapturing()) {
            std::cerr << "Failed to open camera quitting..\n";
            return 1;
        }
        // Read in camera input and save it to the buffer
        std::vector<uint64_t> timestamps;

        // Pausing feature
        bool pause = true;
        std::cerr << "Note: paused, press space to begin recording.\n";
        std::ofstream timestamp_ofs(timestamp_path.string());

        int currFrame = 0;  // current frame number (since launch/last pause)
        while (true) {
            // get latest image from the camera
            cv::Mat xyzMap = camera->getXYZMap();
            cv::Mat rgbMap = camera->getRGBMap();

            if (!xyzMap.empty() && !rgbMap.empty()) {
                ++currFrame;
                if (pause) {
                    const cv::Scalar RECT_COLOR = cv::Scalar(0, 160, 255);
                    const std::string NO_SIGNAL_STR = "PAUSED";
                    const cv::Point STR_POS(rgbMap.cols / 2 - 50,
                                            rgbMap.rows / 2 + 7);
                    const int RECT_WID = 120, RECT_HI = 40;
                    cv::Rect rect(rgbMap.cols / 2 - RECT_WID / 2,
                                  rgbMap.rows / 2 - RECT_HI / 2, RECT_WID,
                                  RECT_HI);

                    // show 'paused' and do not record
                    cv::rectangle(rgbMap, rect, RECT_COLOR, -1);
                    cv::putText(rgbMap, NO_SIGNAL_STR, STR_POS, 0, 0.8,
                                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                    // cv::rectangle(xyzMap, rect, RECT_COLOR / 255.0, -1);
                    // cv::putText(xyzMap, NO_SIGNAL_STR, STR_POS, 0, 0.8,
                    // cv::Scalar(1.0f, 1.0f, 1.0f), 1, cv::LINE_AA);
                } else {
                    // store images
                    auto ts = camera->getTimestamp();
                    if (timestamps.size() && ts - timestamps.back() < 100000)
                        continue;

                    int img_index = timestamps.size();
                    std::stringstream ss_img_id;
                    ss_img_id << std::setw(4) << std::setfill('0')
                              << std::to_string(img_index);
                    const std::string depth_img_path =
                        (depth_path / ("depth_" + ss_img_id.str() + ".depth"))
                            .string();
                    const std::string rgb_img_path =
                        (rgb_path / ("rgb_" + ss_img_id.str() + ".jpg"))
                            .string();
                    cv::Mat depth;
                    cv::extractChannel(xyzMap, depth, 2);

                    // std::cout << "Writing " << depth_img_path <<
                    // std::endl; cv::imwrite(depth_img_path, depth);
                    ark::util::writeDepth(depth_img_path, depth);
                    // std::cout << "Writing " << rgb_img_path << std::endl;
                    cv::imwrite(rgb_img_path, rgbMap);
                    timestamp_ofs << ts << "\n";  // write timestamp

                    // #ifdef AZURE_KINECT_ENABLED
                    // timestamps from camera only supported on Azure Kinect
                    // for now
                    timestamps.push_back(ts);
                    // #else
                    //                     // use system time for other
                    //                     cameras auto curr_time =
                    //                     std::chrono::high_resolution_clock::now();
                    //                     timestamps.push_back(
                    //                             std::chrono::duration_cast<std::chrono::nanoseconds>(curr_time
                    //                             -
                    //                             capture_start_time).count());
                    // #endif
                }
                // visualize
                cv::Mat visual, rgbMapFloat;
                rgbMap.convertTo(rgbMapFloat, CV_32FC3, 1. / 255.);
                cv::hconcat(xyzMap, rgbMapFloat, visual);
                const int MAX_ROWS = 380;
                if (visual.rows > MAX_ROWS) {
                    cv::resize(visual, visual,
                               cv::Size(MAX_ROWS * visual.cols / visual.rows,
                                        MAX_ROWS));
                }
                cv::imshow(camera->getModelName() + " XYZ/RGB Maps", visual);
            }

            int c = cv::waitKey(1);

            // make case insensitive (convert to upper)
            if (c >= 'a' && c <= 'z') c &= 0xdf;

            // 27 is ESC
            if (c == 'Q' || c == 27) {
                break;
            } else if (c == ' ') {
                pause = !pause;
            }
        }
        camera->endCapture();
        cv::destroyWindow(camera->getModelName() + " XYZ/RGB Maps");
        std::cout << "Quitting" << std::endl;

        CameraIntrin intrin;
        // Fit intrinsics from an XYZ map
        // intrin._setVec4d(util::getCameraIntrinFromXYZ(lastXYZMap));
        // Get intrinsics from camera
        intrin = camera->getIntrinsics();

        // Write intrinsics
        intrin.writeFile(intrin_path.string());
        std::cout << "Wrote intrinsics" << std::endl;
    }

    if (verify) {
        std::cout << "Verifying.." << std::endl;
        // To make sure data is good, we will load it from disk
        std::vector<std::string> rgb_files;

        if (is_directory(rgb_path)) {
            boost::filesystem::directory_iterator end_iter;
            for (boost::filesystem::directory_iterator dir_itr(rgb_path);
                 dir_itr != end_iter; ++dir_itr) {
                const auto& next_path = dir_itr->path().generic_string();
                rgb_files.emplace_back(next_path);
            }
            std::sort(rgb_files.begin(), rgb_files.end());
        }
        std::vector<std::string> depth_files;

        if (is_directory(depth_path)) {
            boost::filesystem::directory_iterator end_iter;
            for (boost::filesystem::directory_iterator dir_itr(depth_path);
                 dir_itr != end_iter; ++dir_itr) {
                const auto& next_path = dir_itr->path().generic_string();
                depth_files.emplace_back(next_path);
            }
            std::sort(depth_files.begin(), depth_files.end());
        }
        _ARK_ASSERT_EQ(depth_files.size(), rgb_files.size());

        CameraIntrin intrin;
        intrin.readFile(intrin_path.string());
        _ARK_ASSERT_LT(0.f, intrin.fx);
        std::cout << "Verified" << std::endl;
    }
}
