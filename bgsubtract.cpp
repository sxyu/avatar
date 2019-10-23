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

#include "BGSubtractor.h"
#include "Calibration.h"
#include "Util.h"

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    std::string datasetPath;
    int bgId, imId, padSize;

    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK Synthetic Avatar Depth Image Dataset Generator v0.1b (c) Alex Yu 2019\nPosition arguments");
    po::options_description descCombined("");
    desc.add_options()
        ("help", "produce help message")
        ("background,b", po::value<int>(&bgId)->default_value(0), "Background image id")
        ("image,i", po::value<int>(&imId)->default_value(1), "Current image id")
        ("pad,p", po::value<int>(&padSize)->default_value(4), "Zero pad width for image names in this dataset")
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

    ark::BGSubtractor bgsub(background);
    std::vector<std::array<int, 2> > compsBySize;

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
        if (image.channels() == 1) image = intrin.depthToXYZ(image);
        cv::Mat sub = bgsub.run(image, &compsBySize);

        std::vector<int> colorid(256, 255);
        for (int r = 0 ; r < compsBySize.size(); ++r) {
            colorid[compsBySize[r][1]] = r;// > 0 ? 255 : 0;
        }
        cv::Mat vis(sub.size(), CV_8UC3);
        for (int r = 0 ; r < image.rows; ++r) {
            auto* outptr = vis.ptr<cv::Vec3b>(r); 
            const auto* inptr = sub.ptr<uint8_t>(r);
            for (int c = 0 ; c < image.cols; ++c) {
                int colorIdx = colorid[inptr[c]];
                if (colorIdx >= 254) outptr[c] = 0;
                else outptr[c] = ark::util::paletteColor(colorIdx, true);
            }
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
