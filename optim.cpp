#include <iostream>
#include <boost/program_options.hpp>
#include <boost/thread.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/visualization/pcl_visualizer.h>

#include "AvatarRenderer.h"
#include "AvatarOptimizer.h"
#include "AvatarPCL.h"
#include "Config.h"
#include "Util.h"

namespace {
    using namespace ark;
}

int main(int argc, char** argv) {
    namespace po = boost::program_options;

    std::string intrinPath;
    int numThreads, iters;
    double betaPose, betaShape;
    cv::Size size;

    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK Avatar Optimizer validator (c) Alex Yu 2019\nPosition arguments");
    po::options_description descCombined("");
    desc.add_options()
        ("help", "produce help message")
        (",j", po::value<int>(&numThreads)->default_value(boost::thread::hardware_concurrency()), "Number of threads")
        (",i", po::value<int>(&iters)->default_value(100), "Number of iterations to run")
        ("betapose", po::value<double>(&betaPose)->default_value(0.1), "Cost function weight betaPose")
        ("betashape", po::value<double>(&betaShape)->default_value(0.8), "Cost function weight betaShape")
    ;

    descPositional.add_options()
        ("intrin_path", po::value<std::string>(&intrinPath)->default_value(""), "Path to camera intrinsics file (default: uses hardcoded K4A intrinsics)")
        ("width", po::value<int>(&size.width)->default_value(1280), "Width of generated images")
        ("height", po::value<int>(&size.height)->default_value(720), "Height of generated imaes")
        ;

    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;
    
    po::positional_options_description posopt;
    posopt.add("intrin_path", 1);

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

    const AvatarModel model;
    Avatar ava(model);
    
    Eigen::Vector3d pos;
    pos.x() = random_util::uniform(-1.0, 1.0);
    pos.y() = random_util::uniform(-0.5, 0.5);
    pos.z() = random_util::uniform(2.2, 4.5);
    // pos.z() = 2.0;
    ava.randomize();
    //ava.r[0] = Eigen::AngleAxisd(M_PI, Eigen::Vector3d(0.,1.,0.));
    //ava.r[ark::SmplJoint::R_SHOULDER] = Eigen::AngleAxisd(M_PI/6, Eigen::Vector3d(0.,1.,0.));
    ava.p = pos;
    ava.w.setZero();

    ava.update();

    CameraIntrin intrin;
    if (!intrinPath.empty()) intrin.readFile(intrinPath);
    else {
        intrin.clear();
        intrin.fx = 606.438;
        intrin.fy = 606.351;
        intrin.cx = 637.294;
        intrin.cy = 366.992;
    }

    // BEGIN MAIN PROGRAM
    constexpr char WIND_NAME[] = "Result";
    AvatarRenderer renderer(ava, intrin);
    cv::Mat depth = renderer.renderDepth(size);
    std::vector<Vec3f> vecs;
    cv::Point2f pt;
    for (int r = 0; r < depth.rows; ++r) {
        pt.y = r;
        auto* ptr = depth.ptr<float>(r);
        for (int c = 0; c < depth.cols; ++c) {
            pt.x = c;
            if (ptr[c] <= 0.0) continue;
            vecs.push_back(intrin.to3D(pt, ptr[c]));
        }
    }
    CloudType dataCloud(3, vecs.size());
    for(int i = 0; i < dataCloud.cols(); ++i) {
        dataCloud(0, i) = vecs[i][0];
        dataCloud(1, i) = -vecs[i][1];
        dataCloud(2, i) = vecs[i][2];
    }

    Avatar ava2(model);
    ava2.r = ava.r;

    auto fromSpherical = [](double rho, double theta,
                        double phi, Eigen::Vector3d& out) {
        out[0] = rho * sin(phi) * cos(theta);
        out[1] = rho * cos(phi);
        out[2] = rho * sin(phi) * sin(theta);
    };
    for (int i = 0; i < ava.model.numJoints(); ++i) {
        double theta = random_util::uniform(0, 2 * M_PI);
        double phi   = random_util::uniform(-M_PI/2, M_PI/2);
        Eigen::Vector3d axis_perturb;
        fromSpherical(1.0, theta, phi, axis_perturb);
        double angle_perturb = random_util::randn(0.0, 0.1);
        Eigen::AngleAxisd aa_perturb(angle_perturb, axis_perturb);
        ava2.r[i] *= aa_perturb.toRotationMatrix();
    }
    ava2.p = ava.p;
    ava2.w.setZero();
    ava2.w[0] = -2.5;
    //ava2.r[0] = Eigen::AngleAxisd(M_PI, Eigen::Vector3d(0.,1.,0.));
    //ava2.r[ark::SmplJoint::R_SHOULDER] = Eigen::AngleAxisd(M_PI/6, Eigen::Vector3d(0.,1.,0.));
    ava2.update();

    AvatarOptimizer optim(ava2, intrin, size);
    optim.betaPose = betaPose;
    optim.betaShape = betaShape;
    optim.optimize(dataCloud, iters, numThreads);

    ava2.update();
    
    // auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewport"));
    // auto cloud = ark::avatar_pcl::getCloud(ava);
    // auto mesh2 = ark::avatar_pcl::getMesh(ava2);
    // viewer->setBackgroundColor(0, 0, 0);
    // viewer->addPointCloud(cloud, "cloud", 0);
    // viewer->addPolygonMesh(*mesh2, "meshes-final",0);
    // viewer->addCoordinateSystem(1.0);
    // viewer->initCameraParameters();
    // viewer->spin();

    //cv::namedWindow(WIND_NAME);
    //cv::imshow(WIND_NAME, depth);
    //cv::imshow(WIND_NAME, depth2);
    //cv::waitKey(0);
    // cv::destroyAllWindows();

    return 0;
}
