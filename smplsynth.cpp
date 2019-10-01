#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <chrono>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <boost/lockfree/queue.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <boost/smart_ptr.hpp>
//#include <pcl/visualization/pcl_visualizer.h>

#include <boost/thread.hpp>

#include "AvatarRenderer.h"
#include "Config.h"
#include "Util.h"

namespace {
using namespace ark;

void run(int num_threads, int num_to_gen, std::string out_path, const cv::Size& image_size, const CameraIntrin& intrin, int starting_number, bool overwrite, bool preload)
{
    // Load first joint assignments
    using boost::filesystem::path;
    path outPath (out_path);
    path intrinPath = outPath / "intrin.txt";
    path partMaskPath = outPath / "part_mask";
    path depthPath = outPath / "depth_exr";
    path jointsPath = outPath / "joint";

    if (!boost::filesystem::exists(outPath)) {
        boost::filesystem::create_directories(outPath);
    }
    if (!boost::filesystem::exists(depthPath)) {
        boost::filesystem::create_directories(depthPath);
    }
    if (!boost::filesystem::exists(jointsPath)) {
        boost::filesystem::create_directories(jointsPath);
    }
    if (!boost::filesystem::exists(partMaskPath)) {
        boost::filesystem::create_directories(partMaskPath);
    }

    intrin.writeFile(intrinPath.string());

    boost::lockfree::queue<int> que;
    for (int i = starting_number; i < starting_number+num_to_gen; ++i) {
        if (!overwrite) {
            std::stringstream ss_img_id;
            ss_img_id << std::setw(8) << std::setfill('0') << std::to_string(i);
            auto depthFilePath = 
                depthPath / ("depth_" + ss_img_id.str() + ".exr");
            std::ifstream testIfs(depthFilePath.string());
            if (testIfs) {
                continue;
            }
        }
        que.push(i);
    }

    const AvatarModel model;
    AvatarPoseSequence poseSequence;
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

    auto worker = [&]() {
        Avatar ava(model);
        if (!model.hasPosePrior()) {
            std::cerr << "ERROR: Pose prior required! Please get a version of avatar data with pose_prior.txt\n";
            return;
        }
        if (!model.hasMesh()) {
            std::cerr << "ERROR: Mesh required! Please get a version of avatar data with mesh.txt\n";
            return;
        }

        while(true) {
            int i;
            if (!que.pop(i)) break;
            std::stringstream ss_img_id;
            ss_img_id << std::setw(8) << std::setfill('0') << std::to_string(i);

            if (poseSequence.numFrames) {
                // random_util::randint<size_t>(0, poseSequence.numFrames - 1)
                poseSequence.poseAvatar(ava, i % poseSequence.numFrames);
                ava.r[0].setIdentity();
                ava.randomize(false, true, true);
            } else {
                ava.randomize();
            }
            ava.update();
            
            ark::AvatarRenderer renderer(ava, intrin);

            const std::string depthImgPath = (depthPath / ("depth_" + ss_img_id.str() + ".exr")).string();
            cv::imwrite(depthImgPath, renderer.renderDepth(image_size));
            std::cout << "Wrote " << depthImgPath << std::endl;

            const std::string partMaskImgPath = (partMaskPath / ("part_mask_" + ss_img_id.str() + ".tiff")).string();
            cv::imwrite(partMaskImgPath, renderer.renderPartMask(image_size, part_map::SMPL_JOINT_TO_PART_MAP));
            //std::cout << "Wrote " << partMaskImgPath << std::endl;

            // Output labels
            const std::vector<cv::Point2f>& joints = renderer.getProjectedJoints();
            std::vector<cv::Point2i> jointsi;
            for (auto& pt : joints) jointsi.emplace_back(std::round(pt.x), std::round(pt.y));
            const std::string jointFilePath = (jointsPath / ("joint_" + ss_img_id.str() + ".yml")).string();
            cv::FileStorage fs3(jointFilePath, cv::FileStorage::WRITE);
            fs3 << "joints" << jointsi;

            // Also write xyz positions
            std::vector<cv::Point3f> jointsXYZ;
            for (auto i = 0; i < model.numJoints(); ++i) {
                auto pt = ava.jointPos.col(i);
                jointsXYZ.emplace_back(pt.x(), pt.y(), pt.z());
            }
            fs3 << "joints_xyz" << jointsXYZ;

            // Also write OpenARK avatar parameters
            cv::Point3f p(ava.p(0), ava.p(1), ava.p(2));
            fs3 << "pos" << p;

            std::vector<double> w(model.numShapeKeys());
            std::copy(ava.w.data(), ava.w.data() + w.size(), w.begin());
            fs3 << "shape" << w;

            std::vector<double> r(model.numJoints() * 3);
            for (size_t i = 0; i < ava.r.size(); ++i) {
                Eigen::AngleAxisd aa;
                aa.fromRotationMatrix(ava.r[i]);
                Eigen::Map<Eigen::Vector3d> mp(&r[0] + i*3);
                mp = aa.axis() * aa.angle();
            }
            fs3 << "rots" << r;
            
            // Convert to SMPL parameters
            Eigen::VectorXd smplParams = ava.smplParams();
            std::vector<double> smplParamsVec(smplParams.rows());
            std::copy(smplParams.data(), smplParams.data() + smplParams.rows(), smplParamsVec.begin());
            fs3 << "smpl_params" << smplParamsVec;

            fs3.release();
            // std::cout << "Wrote " << jointFilePath << std::endl;
        }
    };
    std::vector<boost::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }
}
}

int main(int argc, char** argv) {
    namespace po = boost::program_options;

    std::string outPath, intrinPath;
    int startingNumber, numToGen, numThreads;
    bool overwrite, preload;
    cv::Size size;

    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK Synthetic Avatar Depth Image Dataset Generator v0.1b (c) Alex Yu 2019\nPosition arguments");
    po::options_description descCombined("");
    desc.add_options()
        ("help", "produce help message")
        ("overwrite,o", po::bool_switch(&overwrite), "If specified, overwrites existing files. Else, skips over them.")
        ("preload,p", po::bool_switch(&preload), "If specified, pre-loads mocap sequence (if available); WARNING: may take > 5 GB of memory.")
        (",j", po::value<int>(&numThreads)->default_value(boost::thread::hardware_concurrency()), "Number of threads")
    ;

    descPositional.add_options()
        ("output_path", po::value<std::string>(&outPath)->required(), "Output Path")
        ("num_to_gen", po::value<int>(&numToGen)->default_value(1), "Number of images to generate")
        ("starting_number", po::value<int>(&startingNumber)->default_value(0), "Number of images to generate")
        ("intrin_path", po::value<std::string>(&intrinPath)->default_value(""), "Path to camera intrinsics file (default: uses hardcoded K4A intrinsics)")
        ("width", po::value<int>(&size.width)->default_value(1280), "Width of generated images")
        ("height", po::value<int>(&size.height)->default_value(720), "Height of generated imaes")
        ;
    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
    posopt.add("output_path", 1);
    posopt.add("num_to_gen", 1);
    posopt.add("starting_number", 1);
    posopt.add("intrin_path", 1);
    posopt.add("width", 1);
    posopt.add("height", 1);
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

    CameraIntrin intrin;
    if (!intrinPath.empty()) intrin.readFile(intrinPath);
    else {
        intrin.clear();
        intrin.fx = 606.438;
        intrin.fy = 606.351;
        intrin.cx = 637.294;
        intrin.cy = 366.992;
    }

    // auto viewer = boost::make_shared<pcl::visualization::PCLVisualizer>("3D Viewport");
    // HumanAvatar ava("");
    // auto mesh = ark::avatar_pcl::getMesh(ava);
    // viewer->setBackgroundColor(0, 0, 0);
    // viewer->addPolygonMesh(*mesh, "meshes",0);
    // viewer->addCoordinateSystem(1.0);
    // viewer->initCameraParameters();
    // viewer->spin();

    run(numThreads, numToGen, outPath, size, intrin,
           startingNumber, overwrite, preload);
    return 0;
}
