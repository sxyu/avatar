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

//#define DEBUG_PROPOSE
//#define DEBUG

constexpr char WIND_NAME[] = "Result";

namespace {
using namespace ark;

namespace random_util {
template<class T>
/** xorshift-based PRNG */
inline T randint(T lo, T hi) {
    if (hi <= lo) return lo;
    static unsigned long x = std::random_device{}(), y = std::random_device{}(), z = std::random_device{}();
    unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;
    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;
    return z % (hi - lo + 1) + lo;
}

inline double uniform(double min_inc = 0., double max_exc = 1.) {
    thread_local static std::mt19937 rg(std::random_device{}());
    std::uniform_real_distribution<double> normal(min_inc, max_exc); 
    return normal(rg); 
}

inline double randn(double mean = 0, double variance = 1) {
    thread_local static std::mt19937 rg(std::random_device{}());
    std::normal_distribution<double> normal(mean, variance); 
    return normal(rg); 
}

} // random_util

typedef nanoflann::KDTreeEigenMatrixAdaptor<HumanAvatar::EigenCloud2d_T, 2, nanoflann::metric_L2_Simple> kdTree;

void getJoints(HumanAvatar& ava,
               const CameraIntrin& intrin,
               std::vector<cv::Point2i>& joints_out) {
    for (auto i = 0; i < HumanAvatar::NUM_JOINTS; ++i) {
        auto pt = ava.getJointPosition(i);
        joints_out.emplace_back(
                pt.x() * intrin.fx / pt.z() + intrin.cx,
               -pt.y() * intrin.fy / pt.z() + intrin.cy
            );
    }
}

inline void _fromSpherical(double rho, double theta,
                    double phi, Eigen::Vector3d& out) {
    out[0] = rho * sin(phi) * cos(theta);
    out[1] = rho * cos(phi);
    out[2] = rho * sin(phi) * sin(theta);
}

void randomizeParams(HumanAvatar& ava, double shape_sigma = 1.0) {
    // Shape keys
    Eigen::template Map<Eigen::Matrix<double, HumanAvatar::NUM_SHAPEKEYS, 1> > w(ava.w());
    for (int i = 0; i < HumanAvatar::NUM_SHAPEKEYS; ++i) {
        w(i) = random_util::randn() * shape_sigma;
    }

    // Pose
    for (int c = 0; c < ava.posePrior.nComps; ++c) {
        Eigen::Matrix<double, (HumanAvatar::NUM_JOINTS-1) * 3, 1> r;
        for (int i = 0; i < (HumanAvatar::NUM_JOINTS-1) * 3; ++i) {
            r(i) = random_util::randn();
        }
        r *= ava.posePrior.cov_cho[c];
        r += ava.posePrior.mean.row(c);
        for (int i = 0; i < (HumanAvatar::NUM_JOINTS-1); ++i) {
            auto joint = ava.getJoint(i + 1);
            Eigen::AngleAxisd angle_axis;
            angle_axis.angle() = r.segment<3>(i*3).norm();
            angle_axis.axis() = r.segment<3>(i*3)/angle_axis.angle();
            joint->rotation = angle_axis;
        }
    }
    for (int c = 0; c < ava.posePrior.nComps; ++c) {
        Eigen::Matrix<double, (HumanAvatar::NUM_JOINTS-1) * 3, 1> r;
        for (int i = 0; i < (HumanAvatar::NUM_JOINTS-1) * 3; ++i) {
            r(i) = random_util::randn();
        }
        r *= ava.posePrior.cov_cho[c];
        r += ava.posePrior.mean.row(c);
        for (int i = 0; i < (HumanAvatar::NUM_JOINTS-1); ++i) {
            auto joint = ava.getJoint(i + 1);
            Eigen::AngleAxisd angle_axis;
            angle_axis.angle() = r.segment<3>(i*3).norm();
            angle_axis.axis() = r.segment<3>(i*3)/angle_axis.angle();
            joint->rotation = angle_axis;
        }
    }

    // Root position
    Eigen::Vector3d pos;
    pos.x() = random_util::uniform(-1.0, 1.0);
    pos.y() = random_util::uniform(-0.5, 0.5);
    pos.z() = random_util::uniform(2.2, 4.5);
    ava.setCenterPosition(pos);

    // Root rotation
    const Eigen::Vector3d axis_up(0., 1., 0.);
    double angle_up  = random_util::uniform(0.0, std::sqrt(2 * PI));
    angle_up *= angle_up;
    Eigen::AngleAxisd aa_up(angle_up, axis_up);

    double theta = random_util::uniform(0, 2 * PI);
    double phi   = random_util::uniform(-PI/2, PI/2);
    Eigen::Vector3d axis_perturb;
    _fromSpherical(1.0, theta, phi, axis_perturb);
    double angle_perturb = random_util::randn(0.0, 0.2);
    Eigen::AngleAxisd aa_perturb(angle_perturb, axis_perturb);

    ava.setCenterRotation(aa_perturb * aa_up);
}

inline void paintTriangleBary(
        cv::Mat& output_depth,
        const cv::Size& image_size,
        const std::vector<cv::Point2d>& projected,
        const std::vector<HumanAvatar::Point_T,
                          Eigen::aligned_allocator<HumanAvatar::Point_T>>&
                            model_points,
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
    const auto az = model_points[face[xf[0].second]].z,
          bz = model_points[face[xf[1].second]].z,
          cz = model_points[face[xf[2].second]].z;

    int minxi = std::max<int>(a.x, 0),
        maxxi = std::min<int>(c.x, image_size.width-1),
        midxi = std::floor(b.x);

    double denom = 1.0 / ((b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y));
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

            double w1v = (b.y - c.y) * (i - c.x);
            double w2v = (c.y - a.y) * (i - c.x);
            for (int j = minyi; j <= maxyi; ++j) {
                float w1 = (w1v + (c.x - b.x) * (j - c.y)) * denom;
                float w2 = (w2v + (a.x - c.x) * (j - c.y)) * denom;
                output_depth.at<float>(j, i) =
                    w1 * az + w2 * bz + (1. - w1 - w2) * cz;
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
                float w1 = (w1v + (c.x - b.x) * (j - c.y)) * denom;
                float w2 = (w2v + (a.x - c.x) * (j - c.y)) * denom;
                output_depth.at<float>(j, i) =
                    w1 * az + w2 * bz + (1. - w1 - w2) * cz;
            }
        }
    }
}

void run(int num_threads, int num_to_gen, std::string out_path, const cv::Size& image_size, const CameraIntrin& intrin, int starting_number, bool overwrite)
{
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

    using boost::filesystem::path;
    path outPath (out_path);
    path intrinPath = outPath / "intrin.txt";
    path depthPath = outPath / "depth_exr";
    path jointsPath = outPath / "joint";

    if (!boost::filesystem::exists(outPath)) {
        boost::filesystem::create_directories(outPath);
    }
    if (!boost::filesystem::exists(depthPath)) {
        boost::filesystem::create_directories(depthPath);
    } if (!boost::filesystem::exists(jointsPath)) {
        boost::filesystem::create_directories(jointsPath);
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

    typedef std::pair<float, cv::Vec3i> FaceType;
    auto faceComp =  [](const FaceType& a, const FaceType& b) {
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

            randomizeParams(ava);

            ava.update();

            auto modelCloud = ava.getCloud();
            auto& modelPoints = modelCloud->points;

            // Render depth
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

            // Paint the faces using Barycentric coordinate interpolation
            cv::Mat renderedDepth = cv::Mat::zeros(image_size, CV_32F);
            for (int i = 0; i < nFaces;++i) {
                paintTriangleBary(renderedDepth, image_size, projected, modelPoints, faces[i].second);
            }
            //cv::Mat visual;
            //cv::normalize(renderedDepth, visual, 0.0, 1.0, cv::NORM_MINMAX);
            //visual.convertTo(visual, CV_8UC1, 255.0);
            //cv::applyColorMap(visual, visual, cv::COLORMAP_HOT);
            //cv::imshow(WIND_NAME, visual);
            // cv::waitKey(0);

            const std::string depthImgPath = (depthPath / ("depth_" + ss_img_id.str() + ".exr")).string();
            cv::imwrite(depthImgPath, renderedDepth);
            cout << "Wrote " << depthImgPath << endl;

            // Output labels
            std::vector<cv::Point2i> joints;
            getJoints(ava, intrin, joints);

            const std::string jointFilePath = (jointsPath / ("joint_" + ss_img_id.str() + ".yml")).string();
            cv::FileStorage fs3(jointFilePath, cv::FileStorage::WRITE);
            fs3 << "joints" << joints;

            // Also write xyz positions
            std::vector<cv::Point3f> jointsXYZ;
            for (auto i = 0; i < HumanAvatar::NUM_JOINTS; ++i) {
                auto pt = ava.getJointPosition(i);
                jointsXYZ.emplace_back(pt.x(), pt.y(), pt.z());
            }
            fs3 << "joints_xyz" << jointsXYZ;

            // Also write OpenARK avatar parameters
            cv::Point3d p(ava.p()[0], ava.p()[1], ava.p()[2]);
            fs3 << "pos" << p;

            std::vector<double> w(HumanAvatar::NUM_SHAPEKEYS);
            std::copy(ava.w(), ava.w() + w.size(), w.begin());
            fs3 << "shape" << w;

            std::vector<double> r(HumanAvatar::NUM_JOINTS * HumanAvatar::NUM_ROT_PARAMS);
            std::copy(ava.r(), ava.r() + r.size(), r.begin());
            fs3 << "rots" << r;
            
            // Convert to SMPL parameters
            Eigen::VectorXd smplParams = ava.smplParams();
            std::vector<double> smplParamsVec(smplParams.rows());
            std::copy(smplParams.data(), smplParams.data() + smplParams.rows(), smplParamsVec.begin());
            fs3 << "smpl_params" << smplParamsVec;

            fs3.release();
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
    po::options_description descPositional("OpenARK Synthetic Avatar Depth Image Dataset Generator (c) Alex Yu 2019\nPosition arguments");
    po::options_description descCombined("");
    desc.add_options()
        ("help", "produce help message")
        ("overwrite,o", po::bool_switch(&overwrite), "If specified, overwrites existing files. Else, skips over them.")
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
