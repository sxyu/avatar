#include <iostream>
#include <boost/program_options.hpp>

#include "RTree.h"

namespace {
constexpr char WIND_NAME[] = "Image";
}

int main(int argc, char** argv) { 
    std::string data_path, output_path;
    bool verbose;
    int num_threads, num_images, num_points_per_image, num_features, max_probe_offset, min_samples, max_tree_depth;

    namespace po = boost::program_options;
    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK random tree/forest training tool v0.1 (c) Alex Yu 2019\nPositional arguments");
    po::options_description descCombined("");

    desc.add_options()
        ("help", "Produce help message")
        ("output,o", po::value<std::string>(&output_path)->default_value("output.rtree"), "Output file")
        ("threads,j", po::value<int>(&num_threads)->default_value(std::thread::hardware_concurrency()), "Number of threads")
        ("verbose,v", po::bool_switch(&verbose), "Enable verbose output")
        ("images,i", po::value<int>(&num_images)->default_value(1000), "Number of random images to train on")
        ("pixels,p", po::value<int>(&num_points_per_image)->default_value(2000), "Number of random pixels from each image")
        ("features,f", po::value<int>(&num_features)->default_value(2000), "Number of random features to try per tree node")
        ("probe,b", po::value<int>(&max_probe_offset)->default_value(170), "Maximum probe offset for features")
        ("min_samples,m", po::value<int>(&min_samples)->default_value(1), "Minimum number of samples of a child to declare current node a leaf")
        ("depth,d", po::value<int>(&max_tree_depth)->default_value(20), "Maximum tree depth")
    ;

    descPositional.add_options()
        ("data", po::value<std::string>(&data_path)->required(), "Data directory path")
        ;

    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
    posopt.add("data", 1);

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

    if (min_samples < 1) {
        std::cerr << "WARNING: min_samples (-m) cannot be less than 1, defaulting to 1...\n";
        min_samples = 1;
    }
    ark::RTree rtree(16);
    rtree.train(data_path + "/depth_exr", data_path + "/part_mask", num_threads, verbose, num_images, num_points_per_image,
                num_features, max_probe_offset, min_samples, max_tree_depth);
    rtree.exportFile(output_path);

    return 0;
}
