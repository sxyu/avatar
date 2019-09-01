#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <deque>
#include <map>
#include <chrono>
#include <random>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <pcl/search/impl/search.hpp>
#include <pcl/conversions.h>
#include <boost/lockfree/queue.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include "Avatar.h"
#include "GaussianMixture.h"
#include "Calibration.h"
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

inline float uniform(float min_inc = 0., float max_exc = 1.) {
    thread_local static std::mt19937 rg(std::random_device{}());
    std::uniform_real_distribution<float> uniform(min_inc, max_exc); 
    return uniform(rg); 
}

inline float randn(float mean = 0, float variance = 1) {
    thread_local static std::mt19937 rg(std::random_device{}());
    std::normal_distribution<float> normal(mean, variance); 
    return normal(rg); 
}

} // random_util

void getJoints(HumanAvatar& ava,
               const CameraIntrin& intrin,
               std::vector<cv::Point2i>& joints_out) {
    for (auto i = 0; i < ava.numJoints(); ++i) {
        auto pt = ava.jointPos.col(i);
        joints_out.emplace_back(
                pt.x() * intrin.fx / pt.z() + intrin.cx,
               -pt.y() * intrin.fy / pt.z() + intrin.cy
            );
    }
}

inline void _fromSpherical(float rho, float theta,
                    float phi, Eigen::Vector3f& out) {
    out[0] = rho * sin(phi) * cos(theta);
    out[1] = rho * cos(phi);
    out[2] = rho * sin(phi) * sin(theta);
}

void randomizeParams(HumanAvatar& ava, GaussianMixture& pose_prior, float shape_sigma = 1.0) {
    // Shape keys
    for (int i = 0; i < ava.numShapeKeys(); ++i) {
        ava.w(i) = random_util::randn() * shape_sigma;
    }

    // Pose
    // Pick random GMM component
    float randf = random_util::uniform(0.0f, 1.0f);
    int component;
    for (size_t i = 0 ; i < pose_prior.weight.size(); ++i) {
        randf -= pose_prior.weight[i];
        if (randf <= 0) component = i;
    }
    Eigen::VectorXf r((ava.numJoints()-1)*3);
    // Sample from Gaussian
    for (int i = 0; i < (ava.numJoints()-1) * 3; ++i) {
        r(i) = random_util::randn();
    }
    r *= pose_prior.cov_cho[component];
    r += pose_prior.mean.row(component);
    // To rotation matrix
    for (int i = 0; i < ava.numJoints()-1; ++i) {
        Eigen::AngleAxisf angleAxis;
        angleAxis.angle() = r.segment<3>(i*3).norm();
        angleAxis.axis() = r.segment<3>(i*3)/angleAxis.angle();
        ava.r[i + 1] = angleAxis.toRotationMatrix();
    }

    // Root position
    Eigen::Vector3f pos;
    pos.x() = random_util::uniform(-1.0, 1.0);
    pos.y() = random_util::uniform(-0.5, 0.5);
    pos.z() = random_util::uniform(2.2, 4.5);
    ava.p = pos;

    // Root rotation
    const Eigen::Vector3f axis_up(0., 1., 0.);
    float angle_up  = random_util::uniform(-M_PI / 3., M_PI / 3.) + M_PI;
    Eigen::AngleAxisf aa_up(angle_up, axis_up);

    float theta = random_util::uniform(0, 2 * M_PI);
    float phi   = random_util::uniform(-M_PI/2, M_PI/2);
    Eigen::Vector3f axis_perturb;
    _fromSpherical(1.0, theta, phi, axis_perturb);
    float angle_perturb = random_util::randn(0.0, 0.2);
    Eigen::AngleAxisf aa_perturb(angle_perturb, axis_perturb);

    ava.r[0] = (aa_perturb * aa_up).toRotationMatrix();
}

inline void paintDepthTriangleBary(
        cv::Mat& output_depth,
        const cv::Size& image_size,
        const std::vector<cv::Point2f>& projected,
        const ark::CloudType& model_points,
        const cv::Vec3i& face) {
    std::pair<float, int> xf[3] =
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
    const auto az = model_points(2, face[xf[0].second]),
          bz = model_points(2, face[xf[1].second]),
          cz = model_points(2, face[xf[2].second]);

    int minxi = std::max<int>(a.x, 0),
        maxxi = std::min<int>(c.x, image_size.width-1),
        midxi = std::floor(b.x);

    float denom = 1.0 / ((b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y));
    if (a.x != b.x) {
        float mhi = (c.y-a.y)/(c.x-a.x);
        float bhi = a.y - a.x * mhi;
        float mlo = (b.y-a.y)/(b.x-a.x);
        float blo = a.y - a.x * mlo;
        if (b.y > c.y) {
            std::swap(mlo, mhi);
            std::swap(blo, bhi);
        }
        for (int i = minxi; i <= std::min(midxi, image_size.width-1); ++i) {
            int minyi = std::max<int>(std::floor(mlo * i + blo), 0),
                maxyi = std::min<int>(std::ceil(mhi * i + bhi), image_size.height-1);
            if (minyi > maxyi) continue;

            float w1v = (b.y - c.y) * (i - c.x);
            float w2v = (c.y - a.y) * (i - c.x);
            for (int j = minyi; j <= maxyi; ++j) {
                float w1 = (w1v + (c.x - b.x) * (j - c.y)) * denom;
                float w2 = (w2v + (a.x - c.x) * (j - c.y)) * denom;
                output_depth.at<float>(j, i) =
                    w1 * az + w2 * bz + (1. - w1 - w2) * cz;
            }
        }
    }
    if (b.x != c.x) {
        float mhi = (c.y-a.y)/(c.x-a.x);
        float bhi = a.y - a.x * mhi;
        float mlo = (c.y-b.y)/(c.x-b.x);
        float blo = b.y - b.x * mlo;
        if (b.y > a.y) {
            std::swap(mlo, mhi);
            std::swap(blo, bhi);
        }
        for (int i = std::max(midxi, 0)+1; i <= maxxi; ++i) {
            int minyi = std::max<int>(std::floor(mlo * i + blo), 0),
                maxyi = std::min<int>(std::ceil(mhi * i + bhi), image_size.height-1);
            if (minyi > maxyi) continue;

            float w1v = (b.y - c.y) * (i - c.x);
            float w2v = (c.y - a.y) * (i - c.x);
            for (int j = minyi; j <= maxyi; ++j) {
                float w1 = (w1v + (c.x - b.x) * (j - c.y)) * denom;
                float w2 = (w2v + (a.x - c.x) * (j - c.y)) * denom;
                output_depth.at<float>(j, i) =
                    w1 * az + w2 * bz + (1. - w1 - w2) * cz;
            }
        }
    }
}

inline void paintPartsTriangleNN(
        cv::Mat& output_assigned_joint_mask,
        const cv::Size& image_size,
        const std::vector<cv::Point2f>& projected,
        const std::vector<int>& assigned_joint,
        const cv::Vec3i& face) {
    std::pair<float, int> xf[3] =
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
        float mhi = (c.y-a.y)/(c.x-a.x);
        float bhi = a.y - a.x * mhi;
        float mlo = (b.y-a.y)/(b.x-a.x);
        float blo = a.y - a.x * mlo;
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
        float mhi = (c.y-a.y)/(c.x-a.x);
        float bhi = a.y - a.x * mhi;
        float mlo = (c.y-b.y)/(c.x-b.x);
        float blo = b.y - b.x * mlo;
        if (b.y > a.y) {
            std::swap(mlo, mhi);
            std::swap(blo, bhi);
        }
        for (int i = std::max(midxi, 0)+1; i <= maxxi; ++i) {
            int minyi = std::max<int>(std::floor(mlo * i + blo), 0),
                maxyi = std::min<int>(std::ceil(mhi * i + bhi), image_size.height-1);
            if (minyi > maxyi) continue;

            float w1v = (b.y - c.y) * (i - c.x);
            float w2v = (c.y - a.y) * (i - c.x);
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

void run(int num_threads, int num_to_gen, std::string out_path, const cv::Size& image_size, const CameraIntrin& intrin, int starting_number, bool overwrite)
{
    // Load mesh
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

    // Load first joint assignments
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
        float largestWeight = std::numeric_limits<float>::min();
        for (int j = 0; j < nAssign; ++j) {
            int id; float weight;
            skelIfs >> id >> weight;
            if (weight > largestWeight) {
                largestWeight = weight;
                assignedJoint[i] = id;
            }
        }
    }

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
    auto faceComp = [](const FaceType& a, const FaceType& b) {
        return a.first > b.first;
    };

    auto worker = [&]() {
        HumanAvatar ava(util::resolveRootPath("data/avatar-model"));
        GaussianMixture posePrior;
        posePrior.load(util::resolveRootPath("data/avatar-model/pose_prior.txt"));
        std::vector<cv::Point2f> projected(ava.numPoints());

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

            randomizeParams(ava, posePrior);
            ava.update();

            const auto& modelCloud = ava.cloud;
            // Render depth
            for (size_t i = 0; i < modelCloud.cols(); ++i) {
                const auto& pt = modelCloud.col(i);
                projected[i].x = static_cast<float>(pt(0))
                                    * intrin.fx / pt(2) + intrin.cx;
                projected[i].y = -static_cast<float>(pt(1)) * intrin.fy / pt(2) + intrin.cy;
            } 

            // Sort faces by decreasing center depth
            // so that when painted front faces will cover back faces
            for (int i = 0; i < nFaces;++i) {
                auto& face = faces[i].second;
                faces[i].first =
                    (modelCloud(2, face[0]) + modelCloud(2, face[1]) + modelCloud(2, face[2])) / 3.f;
            }
            std::sort(faces.begin(), faces.end(), faceComp);

            // Paint the faces using Barycentric coordinate interpolation
            cv::Mat renderedDepth = cv::Mat::zeros(image_size, CV_32F);
            for (int i = 0; i < nFaces;++i) {
                paintDepthTriangleBary(renderedDepth, image_size, projected, modelCloud, faces[i].second);
            }
            renderedDepth = cv::max(renderedDepth, 0.0f);
            /*
            cv::Mat visual;
            cv::normalize(renderedDepth, visual, 0.0, 1.0, cv::NORM_MINMAX);
            visual.convertTo(visual, CV_8UC1, 255.0);
            cv::applyColorMap(visual, visual, cv::COLORMAP_HOT);
            cv::imshow(WIND_NAME, visual);
            cv::waitKey(0);
            */

            const std::string depthImgPath = (depthPath / ("depth_" + ss_img_id.str() + ".exr")).string();
            cv::imwrite(depthImgPath, renderedDepth);
            std::cout << "Wrote " << depthImgPath << std::endl;

            // Paint the part mask using nearest neighbors
            cv::Mat partMaskMap = cv::Mat::zeros(image_size, CV_8U);
            partMaskMap.setTo(255);
            for (int i = 0; i < nFaces; ++i) {
                paintPartsTriangleNN(partMaskMap, image_size, projected, assignedJoint, faces[i].second);
            }

            const std::string partMaskImgPath = (partMaskPath / ("part_mask_" + ss_img_id.str() + ".tiff")).string();
            cv::imwrite(partMaskImgPath, partMaskMap);
            //std::cout << "Wrote " << partMaskImgPath << std::endl;

            // Output labels
            std::vector<cv::Point2i> joints;
            getJoints(ava, intrin, joints);

            const std::string jointFilePath = (jointsPath / ("joint_" + ss_img_id.str() + ".yml")).string();
            cv::FileStorage fs3(jointFilePath, cv::FileStorage::WRITE);
            fs3 << "joints" << joints;

            // Also write xyz positions
            std::vector<cv::Point3f> jointsXYZ;
            for (auto i = 0; i < ava.numJoints(); ++i) {
                auto pt = ava.jointPos.col(i);
                jointsXYZ.emplace_back(pt.x(), pt.y(), pt.z());
            }
            fs3 << "joints_xyz" << jointsXYZ;

            // Also write OpenARK avatar parameters
            cv::Point3f p(ava.p(0), ava.p(1), ava.p(2));
            fs3 << "pos" << p;

            std::vector<float> w(ava.numShapeKeys());
            std::copy(ava.w.data(), ava.w.data() + w.size(), w.begin());
            fs3 << "shape" << w;

            std::vector<float> r(ava.numJoints() * 3);
            for (size_t i = 0; i < ava.r.size(); ++i) {
                Eigen::AngleAxisf aa;
                aa.fromRotationMatrix(ava.r[i]);
                Eigen::Map<Eigen::Vector3f> mp(&r[0] + i*3);
                mp = aa.axis() * aa.angle();
            }
            fs3 << "rots" << r;
            
            // Convert to SMPL parameters
            Eigen::VectorXf smplParams = ava.smplParams();
            std::vector<float> smplParamsVec(smplParams.rows());
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
