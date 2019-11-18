
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
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
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
    std::string outPath;
    bool skipRecord;//, jointInference;
    bool forceKinect = false, forceRS2 = false;

    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK Data Recording Tool");
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
    ;

    descPositional.add_options()
        ("output_path", po::value<std::string>(&outPath)->required(), "Output Path")
        ;
    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
    posopt.add("output_path", 1);
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

	printf("CONTROLS:\nQ or ESC to stop recording and begin writing dataset to disk,\nSPACE to start/pause"
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
	} if (!boost::filesystem::exists(rgb_path)) {
		boost::filesystem::create_directories(rgb_path);
	}
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

        std::cerr << "Starting data recording, saving to: " << directory_path.string() << "\n";
// #ifndef AZURE_KINECT_ENABLED
        auto capture_start_time = std::chrono::high_resolution_clock::now();
// #endif

        // turn on the camera
        camera->beginCapture();

        // Read in camera input and save it to the buffer
        std::vector<cv::Mat> xyzMaps;
        std::vector<cv::Mat> rgbMaps;
        std::vector<uint64_t> timestamps;

        // Pausing feature
        bool pause = true;
        std::cerr << "Note: paused, press space to begin recording.\n";
        std::ofstream timestamp_ofs(timestamp_path.string());

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
                    cv::rectangle(xyzMap, rect, RECT_COLOR / 255.0, -1);
                    cv::putText(xyzMap, NO_SIGNAL_STR, STR_POS, 0, 0.8, cv::Scalar(1.0f, 1.0f, 1.0f), 1, cv::LINE_AA);
                }
                else {
                    // store images
                    auto ts = static_cast<AzureKinectCamera*>(camera.get())->getTimestamp();
                    if (timestamps.size() && ts - timestamps.back() < 100000) continue;

                    int img_index = timestamps.size();
                    std::stringstream ss_img_id;
                    ss_img_id << std::setw(4) << std::setfill('0') << std::to_string(img_index);
                    const std::string depth_img_path = (depth_path / ("depth_" + ss_img_id.str() + ".depth")).string();
                    const std::string rgb_img_path = (rgb_path / ("rgb_" + ss_img_id.str() + ".jpg")).string();
                    cv::Mat depth; cv::extractChannel(xyzMap, depth, 2);

                    // xyzMaps.push_back(xyzMap);
                    // rgbMaps.push_back(rgbMap);
                    // std::cout << "Writing " << depth_img_path << std::endl;
                    // cv::imwrite(depth_img_path, depth);
                    writeDepth(depth_img_path, depth);
                    // std::cout << "Writing " << rgb_img_path << std::endl;
                    cv::imwrite(rgb_img_path, rgbMap);
                    timestamp_ofs << ts << "\n"; // write timestamp

// #ifdef AZURE_KINECT_ENABLED
                    // timestamps from camera only supported on Azure Kinect for now
                    timestamps.push_back(ts);
// #else
//                     // use system time for other cameras
//                     auto curr_time = std::chrono::high_resolution_clock::now();
//                     timestamps.push_back(
//                             std::chrono::duration_cast<std::chrono::nanoseconds>(curr_time - capture_start_time).count());
// #endif
                }
                // visualize
                cv::Mat visual, rgbMapFloat;
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

            // 27 is ESC
            if (c == 'Q' || c == 27) {
                break;
            }
            else if (c == ' ') {
                pause = !pause;
            }
        }
        camera->endCapture();
        cv::destroyWindow(camera->getModelName() + " XYZ/RGB Maps");

        // std::cerr <<xyzMaps.size() << " "  << rgbMaps.size() << std::endl;

        // Write the captured frames to disk
        // ARK_ASSERT(xyzMaps.size() == rgbMaps.size(), "Depth map and RGB map are not in sync!");


        // int img_index = 0;
        // for (int i = 0; i < xyzMaps.size(); ++i) {
        //     std::cout << "Writing " << i << " / " << xyzMaps.size() << std::endl;
        //     std::stringstream ss_img_id;
        //     ss_img_id << std::setw(4) << std::setfill('0') << std::to_string(img_index);
        //     const std::string depth_img_path = (depth_path / ("depth_" + ss_img_id.str() + ".depth")).string();
        //     const std::string rgb_img_path = (rgb_path / ("rgb_" + ss_img_id.str() + ".jpg")).string();
        //     cv::Mat depth; cv::extractChannel(xyzMaps[i], depth, 2);
        //     if (!cv::countNonZero(depth)) {
        //         std::cerr << "WARNING: depth image " << i << " is blank, skipping\n"; continue;
        //     }
        //     std::cout << "Writing " << depth_img_path << std::endl;
        //     // cv::imwrite(depth_img_path, depth);
        //     writeDepth(depth_img_path, depth);
        //     std::cout << "Writing " << rgb_img_path << std::endl;
        //     cv::imwrite(rgb_img_path, rgbMaps[i]);
        //     timestamp_ofs << timestamps[i] << "\n"; // write timestamp
        //     ++img_index;
        // }
        timestamp_ofs.close();

        // fit intrinsics from an XYZ map
        intrin = util::getCameraIntrinFromXYZ(xyzMaps[xyzMaps.size()/2]);
        // write intrinsics
        std::ofstream intrin_ofs(intrin_path.string());
        intrin_ofs <<
            "fx " << intrin[0] << "\n" <<
            "cx " << intrin[1] << "\n" <<
            "fy " << intrin[2] << "\n" <<
            "cy " << intrin[3] << "\n";
        intrin_ofs.close();
    }

    // To make sure data is good, we will load it from disk rather than reusing
	// Read in all the rgb images
	std::vector<std::string> rgb_files;

	if (is_directory(rgb_path)) {
		boost::filesystem::directory_iterator end_iter;
		for (boost::filesystem::directory_iterator dir_itr(rgb_path); dir_itr != end_iter; ++dir_itr) {
			const auto& next_path = dir_itr->path().generic_string();
			rgb_files.emplace_back(next_path);
		}
		std::sort(rgb_files.begin(), rgb_files.end());
	}
	std::vector<std::string> depth_files;

	if (is_directory(depth_path)) {
		boost::filesystem::directory_iterator end_iter;
		for (boost::filesystem::directory_iterator dir_itr(depth_path); dir_itr != end_iter; ++dir_itr) {
			const auto& next_path = dir_itr->path().generic_string();
			depth_files.emplace_back(next_path);
		}
		std::sort(depth_files.begin(), depth_files.end());
	}
	ARK_ASSERT(depth_files.size() == rgb_files.size(), "Depth map and RGB map are not in sync!");

    std::ifstream intrin_ifs(intrin_path.string());
    if (intrin_ifs) {
        // depth files are depth maps (require intrinsics to convert)
        std::string _garbage;
        intrin_ifs >> _garbage >> intrin[0];
        intrin_ifs >> _garbage >> intrin[1];
        intrin_ifs >> _garbage >> intrin[2];
        intrin_ifs >> _garbage >> intrin[3];
    }

	cv::destroyAllWindows();
	return 0;
}
