#include <iostream>
#include <boost/program_options.hpp>

#include "RTree.h"
#include "Avatar.h"
#include "Config.h"

namespace {
constexpr char WIND_NAME[] = "Image";
}

int main(int argc, char** argv) {
    std::string input_path, output_path, intrin_path, resume_file;
    bool verbose, preload;
    int num_threads, num_images;
    float frac_samples_per_feature;
    cv::Size size;

    namespace po = boost::program_options;
    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK random tree/forest iterative refinement tool v0.2 (c) Alex Yu 2019\nPositional arguments");
    po::options_description descCombined("");

    desc.add_options()
        ("help", "Produce help message")
        ("output,o", po::value<std::string>(&output_path)->default_value(""), "Output file; default is <input>.refine.srtr")
        ("threads,j", po::value<int>(&num_threads)->default_value(std::thread::hardware_concurrency()), "Number of threads")
        ("verbose,v", po::bool_switch(&verbose), "Enable verbose output")
        ("preload", po::bool_switch(&preload), "Preload avatar pose sequence in memory to speed up random pose; only useful if using synthetic data input")
        ("images,i", po::value<int>(&num_images)->default_value(100), "Number of random images to train on; Kinect used 1 million")
        ("intrin_path", po::value<std::string>(&intrin_path)->default_value(""), "Path to camera intrinsics file (default: uses hardcoded K4A intrinsics)")
        ("width", po::value<int>(&size.width)->default_value(1280), "Width of generated images; only useful if using synthetic data input")
        ("height", po::value<int>(&size.height)->default_value(720), "Height of generated imaes; only useful if using synthetic data input")
    ;
    descPositional.add_options()
        ("tree,r", po::value<std::string>(&input_path)->required(), "Input tree file")
    ;

    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
    posopt.add("tree", 1);

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

    ark::RTree rtree(16);
    ark::AvatarModel model;
    ark::AvatarPoseSequence poseSequence;
    if (poseSequence.numFrames) {
        std::cerr << "Using mocap sequence with " << poseSequence.numFrames << " frames to generate poses\n";
        if (preload) {
            std::cerr << "Pre-loading sequence...\n";
            poseSequence.preload();
            std::cerr << "Pre-loading done\n";
        }
    } else{
        std::cerr << "WARNING: no mocap pose sequence found, will fallback to GMM to generate poses\n";
    }

    ark::CameraIntrin intrin;
    if (!intrin_path.empty()) intrin.readFile(intrin_path);
    else {
        intrin.clear();
        intrin.fx = 606.438;
        intrin.fy = 606.351;
        intrin.cx = 637.294;
        intrin.cy = 366.992;
    }
    if (output_path.empty()) {
        output_path = input_path;
        if (input_path.size() > 5 && !input_path.compare(input_path.size()-5, input_path.size(), ".srtr")) {
            for (int i = 0; i < 5; ++i) output_path.pop_back();
        }
        output_path.append(".refine.srtr");
    }
    rtree.loadFile(input_path);
    rtree.trainTransfer(model, poseSequence, intrin, size, num_threads, verbose, num_images, ark::part_map::SMPL_JOINT_TO_PART_MAP);
    rtree.exportFile(output_path);

    return 0;
}
