#include <fstream>
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <chrono>
#include <random>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
//#include <pcl/surface/concave_hull.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/conversions.h>
#include <boost/lockfree/queue.hpp>
#include <boost/program_options.hpp>

#include "Avatar.h"
#include "Calibration.h"
#include "HumanDetector.h"
#include "Util.h"
//#include "concaveman.h"

//#define DEBUG_PROPOSE
//#define DEBUG

constexpr char WIND_NAME[] = "Result";

namespace {
using namespace ark;
inline void paintTriangleNN(
        cv::Mat& output_assigned_joint_mask,
        const cv::Size& image_size,
        const std::vector<cv::Point2d>& projected,
        const std::vector<int>& assigned_joint,
        const cv::Vec3i& face) {
    std::pair<double, int> xf[3] =
    {
        {projected[face[0]].x, 0},
        {projected[face[1]].x, 1},
        {projected[face[2]].x, 2}
    };
    std::sort(xf, xf+3);

    // reorder points for convenience
    auto a = projected[face[xf[0].second]],
         b = projected[face[xf[1].second]],
         c = projected[face[xf[2].second]];
    a.x = std::floor(a.x);
    c.x = std::ceil(c.x);
    if (a.x == c.x) return;

    auto assigned_a = assigned_joint[face[xf[0].second]],
         assigned_b = assigned_joint[face[xf[1].second]],
         assigned_c = assigned_joint[face[xf[2].second]];
    /*
    const auto az = model_points[face[xf[0].second]].z,
          bz = model_points[face[xf[1].second]].z,
          cz = model_points[face[xf[2].second]].z;
          */

    int minxi = std::max<int>(a.x, 0),
        maxxi = std::min<int>(c.x, image_size.width-1),
        midxi = std::floor(b.x);

    if (a.x != b.x) {
        double mhi = (c.y-a.y)/(c.x-a.x);
        double bhi = a.y - a.x * mhi;
        double mlo = (b.y-a.y)/(b.x-a.x);
        double blo = a.y - a.x * mlo;
        if (b.y > c.y) {
            std::swap(mlo, mhi);
            std::swap(blo, bhi);
        }
        for (int i = minxi; i <= std::min(midxi, image_size.width-1); ++i) {
            int minyi = std::max<int>(std::floor(mlo * i + blo), 0),
                maxyi = std::min<int>(std::ceil(mhi * i + bhi), image_size.height-1);
            if (minyi > maxyi) continue;
            
            for (int j = minyi; j <= maxyi; ++j) {
                auto& out = output_assigned_joint_mask.at<uint8_t>(j, i);
                int dista = (a.x - i) * (a.x - i) + (a.y - j) * (a.y - j);
                int distb = (b.x - i) * (b.x - i) + (b.y - j) * (b.y - j);
                int distc = (c.x - i) * (c.x - i) + (c.y - j) * (c.y - j);
                if (dista < distb && dista < distc) {
                     out = assigned_a;
                } else if (distb < distc) {
                     out = assigned_b;
                } else {
                     out = assigned_c;
                }
            }
        }
    }
    if (b.x != c.x) {
        double mhi = (c.y-a.y)/(c.x-a.x);
        double bhi = a.y - a.x * mhi;
        double mlo = (c.y-b.y)/(c.x-b.x);
        double blo = b.y - b.x * mlo;
        if (b.y > a.y) {
            std::swap(mlo, mhi);
            std::swap(blo, bhi);
        }
        for (int i = std::max(midxi, 0)+1; i <= maxxi; ++i) {
            int minyi = std::max<int>(std::floor(mlo * i + blo), 0),
                maxyi = std::min<int>(std::ceil(mhi * i + bhi), image_size.height-1);
            if (minyi > maxyi) continue;

            double w1v = (b.y - c.y) * (i - c.x);
            double w2v = (c.y - a.y) * (i - c.x);
            for (int j = minyi; j <= maxyi; ++j) {
                auto& out = output_assigned_joint_mask.at<uint8_t>(j, i);
                int dista = (a.x - i) * (a.x - i) + (a.y - j) * (a.y - j);
                int distb = (b.x - i) * (b.x - i) + (b.y - j) * (b.y - j);
                int distc = (c.x - i) * (c.x - i) + (c.y - j) * (c.y - j);
                if (dista < distb && dista < distc) {
                     out = assigned_a;
                } else if (distb < distc) {
                     out = assigned_b;
                } else {
                     out = assigned_c;
                }
            }
        }
    }
}

void run(int num_threads, int max_num_to_gen, std::string out_path, const cv::Size& image_size, const CameraIntrin& intrin, int starting_number, bool overwrite)
{
    // Loadm mesh
    std::ifstream meshIfs(util::resolveRootPath("data/avatar-model/mesh.txt"));
    int nFaces;
    meshIfs >> nFaces;
    std::vector<cv::Vec3i> mesh;
    mesh.reserve(nFaces);
    for (int i = 0; i < nFaces;++i) {
        mesh.emplace_back();
        auto& face = mesh.back();
        meshIfs >> face[0] >> face[1] >> face[2];
    }

    // Load joint assignments
    std::ifstream skelIfs(util::resolveRootPath("data/avatar-model/skeleton.txt"));
    int nJoints, nPoints;
    skelIfs >> nJoints >> nPoints;
    skelIfs.ignore(1000, '\n');
    std::string _;
    for (int i = 0; i < nJoints; ++i)
        std::getline(skelIfs, _);
    std::vector<int> assignedJoint(nPoints);
    for (int i = 0; i < nPoints; ++i) {
        int nAssign;  skelIfs >> nAssign;
        double largestWeight = -DBL_MAX;
        for (int j = 0; j < nAssign; ++j) {
            int id; double weight;
            skelIfs >> id >> weight;
            if (weight > largestWeight) {
                largestWeight = weight;
                assignedJoint[i] = id;
            }
        }
    }

    using boost::filesystem::path;
    path outPath (out_path);
    path partMaskPath = outPath / "part_mask";
    path jointsPath = outPath / "joint";

    if (!boost::filesystem::exists(outPath)) {
        boost::filesystem::create_directories(outPath);
    }
    if (!boost::filesystem::exists(jointsPath)) {
        boost::filesystem::create_directories(jointsPath);
    }
    if (!boost::filesystem::exists(partMaskPath)) {
        boost::filesystem::create_directories(partMaskPath);
    }

    boost::lockfree::queue<int> que;
    for (int i = starting_number; i < starting_number+max_num_to_gen; ++i) {
        std::stringstream ss_img_id;
        ss_img_id << std::setw(8) << std::setfill('0') << std::to_string(i);
        auto jointFilePath = jointsPath / ("joint_" + ss_img_id.str() + ".yml");
        if (!boost::filesystem::exists(jointFilePath.string())) break;
        if (!overwrite) {
            auto partMaskFilePath = 
                partMaskPath / ("part_mask_" + ss_img_id.str() + ".png");
            if (boost::filesystem::exists(partMaskFilePath.string())) {
                continue;
            }
        }
        que.push(i);
    }

    typedef std::pair<float, cv::Vec3i> FaceType;
    auto faceComp = [](const FaceType& a, const FaceType& b) {
        return a.first > b.first;
    };

    auto worker = [&]() {
        HumanAvatar ava(HumanDetector::HUMAN_MODEL_PATH, HumanDetector::HUMAN_MODEL_SHAPE_KEYS);
        std::vector<cv::Point2d> projected(ava.getCloud()->size());

        std::vector<FaceType> faces;
        faces.reserve(nFaces);
        for (int i = 0; i < nFaces;++i) {
            faces.emplace_back(0.f, mesh[i]);
        }

        while(true) {
            int i;
            if (!que.pop(i)) break;
            std::stringstream ss_img_id;
            ss_img_id << std::setw(8) << std::setfill('0') << std::to_string(i);

            std::string jointFilePath = (jointsPath / ("joint_" + ss_img_id.str() + ".yml")).string();

            cv::FileStorage fs2(jointFilePath, cv::FileStorage::READ);
            std::vector<double> p;
            fs2["pos"] >> p;
            std::vector<double> w;
            fs2["shape"] >> w;
            std::vector<double> r;
            fs2["rots"] >> r;
            fs2.release();

            std::copy(r.begin(), r.end(), ava.r());
            std::copy(w.begin(), w.end(), ava.w());
            std::copy(p.begin(), p.end(), ava.p());

            ava.update();

            auto modelCloud = ava.getCloud();
            auto& modelPoints = modelCloud->points;

            // Compute part labels
            for (size_t i = 0; i < modelCloud->size(); ++i) {
                const auto& pt = modelPoints[i];
                projected[i].x = static_cast<double>(pt.x)
                                    * intrin.fx / pt.z + intrin.cx;
                projected[i].y = -static_cast<double>(pt.y) * intrin.fy / pt.z + intrin.cy;
            } 

            // Sort faces by decreasing center depth
            // so that when painted front faces will cover back faces
            for (int i = 0; i < nFaces;++i) {
                auto& face = faces[i].second;
                faces[i].first =
                    (modelPoints[face[0]].z + modelPoints[face[1]].z + modelPoints[face[2]].z) / 3.f;
            }
            std::sort(faces.begin(), faces.end(), faceComp);

            // Paint the faces using nearest neighbors
            cv::Mat partMaskMap(image_size, CV_8U);
            partMaskMap.setTo(255);
            for (int i = 0; i < nFaces;++i) {
                paintTriangleNN(partMaskMap, image_size, projected, assignedJoint, faces[i].second);
            }

            const std::string partMaskImgPath = (partMaskPath / ("part_mask_" + ss_img_id.str() + ".png")).string();
            cv::imwrite(partMaskImgPath, partMaskMap);
            cout << "Wrote " << partMaskImgPath << endl;
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
    bool overwrite;
    cv::Size size;

    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK Synthetic Avatar Dataset Part Label Mask Generator (c) Alex Yu 2019\nPosition arguments");
    po::options_description descCombined("");
    desc.add_options()
        ("help", "produce help message")
        ("overwrite,o", po::bool_switch(&overwrite), "If specified, overwrites existing files. Else, skips over them.")
        (",j", po::value<int>(&numThreads)->default_value(boost::thread::hardware_concurrency()), "Number of threads")
    ;

    descPositional.add_options()
        ("output_path", po::value<std::string>(&outPath)->required(), "Input/Output DataSet Path")
        ("num_to_gen", po::value<int>(&numToGen)->default_value(INT_MAX), "Maximum number of images to generate")
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
    /*
    auto viewer = Visualizer::getPCLVisualizer();
    HumanAvatar ava(HumanDetector::HUMAN_MODEL_PATH, HumanDetector::HUMAN_MODEL_SHAPE_KEYS);
    pcl::PolygonMesh mesh;
    auto cloud = ava.getCloud();
    pcl::toPCLPointCloud2(*cloud, mesh.cloud);
    std::ifstream meshIfs(util::resolveRootPath("data/avatar-model/mesh.txt"));
    int nFaces;
    meshIfs >> nFaces;
    for (int i = 0; i < nFaces;++i) {
        mesh.polygons.emplace_back();
        auto& face = mesh.polygons.back().vertices; 
        face.resize(3);
        meshIfs >> face[0] >> face[1] >> face[2];
    }
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPolygonMesh(mesh,"meshes",0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->spin();
    */

    run(numThreads, numToGen, outPath, size, intrin,
           startingNumber, overwrite);
    return 0;
}
