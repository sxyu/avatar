#include <iostream>
#include <boost/program_options.hpp>

#include "RTree.h"
#include "Avatar.h"
#include "Config.h"

namespace {
constexpr char WIND_NAME[] = "Image";
}

int main(int argc, char** argv) {
    std::string data_path, output_path, intrin_path, resume_file;
    bool verbose, preload, generate_samples_only;
    int num_threads, num_images, num_points_per_image, num_features, num_features_filtered, max_probe_offset, min_samples, max_tree_depth,
        min_samples_per_feature, threshes_per_feature, cache_size;
    float frac_samples_per_feature;
    cv::Size size;

    namespace po = boost::program_options;
    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK random tree/forest training tool v0.2 (c) Alex Yu 2019\nPositional arguments");
    po::options_description descCombined("");

    desc.add_options()
        ("help", "Produce help message")
        ("output,o", po::value<std::string>(&output_path)->default_value("output.rtree"), "Output file")
        ("threads,j", po::value<int>(&num_threads)->default_value(std::thread::hardware_concurrency()), "Number of threads")
        ("verbose,v", po::bool_switch(&verbose), "Enable verbose output")
        ("preload", po::bool_switch(&preload), "Preload avatar pose sequence in memory to speed up random pose; "
                                               "may take multiple GBs of RAM, only useful if using synthetic data input")
        ("images,i", po::value<int>(&num_images)->default_value(100), "Number of random images to train on; Kinect used 1 million")
        ("intrin_path", po::value<std::string>(&intrin_path)->default_value(""), "Path to camera intrinsics file (default: uses hardcoded K4A intrinsics)")
        ("pixels,p", po::value<int>(&num_points_per_image)->default_value(2000), "Number of random pixels from each image; Kinect used 2000")
        ("features,f", po::value<int>(&num_features)->default_value(5000), "Number of random features to try per tree node on sparse samples; Kinect used 2000")
        ("features_filtered,F", po::value<int>(&num_features_filtered)->default_value(200), "Number of random features to try per tree node on dense samples")
        ("probe,b", po::value<int>(&max_probe_offset)->default_value(170), "Maximum probe offset for random feature generation. "
                            "Noted in Kinect paper that cost 'levels off around >=129' but hyperparameter value not provided")
        ("min_samples,m", po::value<int>(&min_samples)->default_value(1), "Minimum number of samples of a child to declare current node a leaf")
        ("min_samples_per_feature", po::value<int>(&min_samples_per_feature)->default_value(20),
          "Minimum number of sparse samples to use in each node training step to quickly propose thresholds. If num_samples * frac_samples_per_feature < min_samples_per_feature then min_samples_per_feature samples are used.")
        ("frac_samples_per_feature", po::value<float>(&frac_samples_per_feature)->default_value(0.001f),
          "Proportion of samples to use in each node training step to sparsely propose thresholds.")
        ("threshes_per_feature", po::value<int>(&threshes_per_feature)->default_value(15),
          "Maximum number of candidates thresholds to optimize over for each feature (different from Kinect)")
        ("depth,d", po::value<int>(&max_tree_depth)->default_value(20), "Maximum tree depth; Kinect used 20")
        ("width", po::value<int>(&size.width)->default_value(1280), "Width of generated images; only useful if using synthetic data input")
        ("height", po::value<int>(&size.height)->default_value(720), "Height of generated imaes; only useful if using synthetic data input")
        ("cache_size,c", po::value<int>(&cache_size)->default_value(50), "Max number of images in cache during training")
        ("resume,s", po::value<std::string>(&resume_file)->default_value(""), "Training save state file (previously known as 'samples' file, now more general).")
        ("gen_samples,g", po::bool_switch(&generate_samples_only), "If specified, skips training and only generates samples file (must specify -s=PATH)")
    ;

    descPositional.add_options()
        ("data", po::value<std::string>(&data_path)->default_value("://SMPLSYNTH"), "Data directory path; leave blank to generate simulated data")
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
    if (data_path == "://SMPLSYNTH") {
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
        rtree.trainFromAvatar(model, poseSequence, intrin, size, num_threads, verbose, num_images, num_points_per_image,
                num_features, num_features_filtered, max_probe_offset, min_samples, max_tree_depth, min_samples_per_feature, frac_samples_per_feature,
                threshes_per_feature, ark::part_map::SMPL_JOINT_TO_PART_MAP, cache_size, resume_file);
    } else {
        rtree.train(data_path + "/depth_exr", data_path + "/part_mask", num_threads, verbose, num_images, num_points_per_image,
                num_features, num_features_filtered, max_probe_offset, min_samples, max_tree_depth, min_samples_per_feature, frac_samples_per_feature, threshes_per_feature,
                cache_size, resume_file);
    }
    rtree.exportFile(output_path);

    return 0;
}
