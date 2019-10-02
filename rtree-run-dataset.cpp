#include <iostream>
#include <boost/program_options.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "RTree.h"

namespace {
constexpr char WIND_NAME[] = "Image";

cv::Vec3b paletteColor(int color_index, bool bgr)
{
    using cv::Vec3b;
    static const Vec3b palette[] = {
        Vec3b(0, 220, 255), Vec3b(177, 13, 201), Vec3b(94, 255, 34),
        Vec3b(54, 65, 255), Vec3b(64, 255, 255), Vec3b(217, 116, 0),
        Vec3b(27, 133, 255), Vec3b(190, 18, 240), Vec3b(20, 31, 210),
        Vec3b(75, 20, 133), Vec3b(255, 219, 127), Vec3b(204, 204, 57),
        Vec3b(112, 153, 61), Vec3b(64, 204, 46), Vec3b(112, 255, 1),
        Vec3b(170, 170, 170), Vec3b(225, 30, 42), Vec3b(255, 255, 32),
        Vec3b(255, 45, 250), Vec3b(101, 0, 209), Vec3b(40, 70, 50),
        Vec3b(100, 100, 100), Vec3b(105, 200, 120), Vec3b(150,150,150)
    };

    if (color_index == 255) return Vec3b(0, 0, 0);
    Vec3b color = palette[color_index % (int)(sizeof palette / sizeof palette[0])];
    return bgr ? color : Vec3b(color[2], color[1], color[0]);
}
}

int main(int argc, char** argv) {

    std::vector<std::string> model_paths;
    std::string image_path;

    namespace po = boost::program_options;
    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK Random Tree/Forest empirical validation tool v0.1 (c) Alex Yu 2019\nPositional arguments");
    po::options_description descCombined("");

    desc.add_options()
        ("help", "Produce help message")
    ;

    descPositional.add_options()
        ("image", po::value<std::string>(&image_path)->required(), "Depth image (.exr) to run model on")
        ("models", po::value<std::vector<std::string> >(&model_paths)->required(), "Model path (from rtree-train)")
        ;

    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
    posopt.add("image", 1);
    posopt.add("models", -1);

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
    std::vector<cv::Mat> result;

    if (model_paths.empty()) {
        std::cerr << "Error: please specify at least one model path" << "\n";
        std::cerr << descPositional << "\n" << desc << "\n";
        return 1;
    }

    cv::Mat image = cv::imread(image_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    for (auto& model_path: model_paths) {
        ark::RTree rtree(model_path);
        std::vector<cv::Mat> model_results = rtree.predict(image);
        if (result.empty()) result = model_results;
        else {
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] += model_results[i];
            }
        }
    }
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] /= model_paths.size();
    }

    cv::Mat maxVals(image.size(), CV_32F);
    maxVals.setTo(0);
    cv::Mat visual = cv::Mat::zeros(image.size(), CV_8UC3);
    for (size_t i = 0; i < result.size(); ++i) {
        for (int r = 0; r < image.rows; ++r) {
            auto* imPtr = image.ptr<float>(r);
            auto* inPtr = result[i].ptr<float>(r);
            auto* maxValPtr = maxVals.ptr<float>(r);
            auto* visualPtr = visual.ptr<cv::Vec3b>(r);
            for (int c = 0; c < image.cols; ++c){
                if (imPtr[c] == 0.0) continue;
                if (inPtr[c] > maxValPtr[c]) {
                    maxValPtr[c] = inPtr[c];
                    visualPtr[c] = paletteColor(i, true);
                }
            }
        }
    }
    cv::imshow(WIND_NAME, visual);
    cv::waitKey(0);

    /*
    for (size_t i = 0; i < result.size(); ++i) {
        std::cerr << i << "\n";
        cv::normalize(result[i], result[i], 0.0, 1.0, cv::NORM_MINMAX);
        cv::imshow(WIND_NAME, result[i]);
        cv::waitKey(0);
    }
    */

    return 0;
}
