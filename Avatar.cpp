#include "Avatar.h"
#include "AvatarPCL.h"
#include "AvatarRenderer.h"

#include <fstream>
#include <chrono>
#include <boost/filesystem.hpp>
#include <boost/smart_ptr.hpp>
#include <pcl/conversions.h>

#include "Util.h"

#define BEGIN_PROFILE auto start = std::chrono::high_resolution_clock::now()
#define PROFILE(x) do{printf("%s: %f ms\n", #x, std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count()); start = std::chrono::high_resolution_clock::now(); }while(false)

namespace {
    /** Hand-written faster function to load a saved PCL point cloud directly
     *  into an Eigen vector, where points are stored: x1 y1 z1 x2 y2 z2 ...
     *  The reason we flatten the cloud instead of using a matrix is to make it easier
     *  to add in shape keys, which would otherwise need to be tensors */
    Eigen::VectorXd loadPCDToPointVectorFast(const std::string& path) {
        std::ifstream pcd(path);
        int nPoints = -1;
        while (pcd) {
            std::string label;
            pcd >> label;
            if (label == "DATA") {
                if (nPoints < 0) { 
                    std::cerr << "ERROR: invalid PCD file at " << path << ": no WIDTH field before data, so "
                                 "we don't know how many points there are!\n";
                    std::exit(0);
                }
                pcd >> label;
                if (label != "ascii") {
                    std::cerr << "ERROR: non-ascii PCD not supported! File " << path << "\n";
                    std::exit(0);
                }
                break;
            } else if (label == "WIDTH") {
                pcd >> nPoints;
                pcd.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            } else {
                std::string _;
                std::getline(pcd, _);
            }
        }
        if (!pcd || nPoints < 0) {
            std::cerr << "ERROR: invalid PCD file at " << path << ": unexpected EOF\n";
            std::exit(0);
        }

        Eigen::VectorXd result(nPoints * 3);
        for (int i = 0; i < nPoints * 3; ++i) {
            pcd >> result(i);
        }
        return result;
    }

    /** Spherical to rectangular coords */
    inline void fromSpherical(double rho, double theta,
                        double phi, Eigen::Vector3d& out) {
        out[0] = rho * sin(phi) * cos(theta);
        out[1] = rho * cos(phi);
        out[2] = rho * sin(phi) * sin(theta);
    }

    /** Paint projected triangle on depth map using barycentric linear interp */
    inline void paintDepthTriangleBary(
            cv::Mat& output_depth,
            const cv::Size& image_size,
            const std::vector<cv::Point2f>& projected,
            const ark::CloudType& model_points,
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
        const auto az = model_points(2, face[xf[0].second]),
              bz = model_points(2, face[xf[1].second]),
              cz = model_points(2, face[xf[2].second]);

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
                    double w1 = (w1v + (c.x - b.x) * (j - c.y)) * denom;
                    double w2 = (w2v + (a.x - c.x) * (j - c.y)) * denom;
                    output_depth.at<float>(j, i) =
                        static_cast<float>(w1 * az + w2 * bz + (1. - w1 - w2) * cz);
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
                    double w1 = (w1v + (c.x - b.x) * (j - c.y)) * denom;
                    double w2 = (w2v + (a.x - c.x) * (j - c.y)) * denom;
                    output_depth.at<float>(j, i) =
                        static_cast<float>(w1 * az + w2 * bz + (1. - w1 - w2) * cz);
                }
            }
        }
    }

    /** Paint projected triangle on part mask (CV_8U) using nearest neighbors interp */
    inline void paintPartsTriangleNN(
            cv::Mat& output_assigned_joint_mask,
            const cv::Size& image_size,
            const std::vector<cv::Point2f>& projected,
            const std::vector<std::vector<std::pair<double, int> > > & assigned_joint,
            const cv::Vec3i& face,
            const int* part_map) {
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

        auto assigned_a = assigned_joint[face[xf[0].second]][0].second,
             assigned_b = assigned_joint[face[xf[1].second]][0].second,
             assigned_c = assigned_joint[face[xf[2].second]][0].second;
        if (part_map != nullptr) {
            assigned_a = part_map[assigned_a];
            assigned_b = part_map[assigned_b];
            assigned_c = part_map[assigned_c];
        }
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

    /** Paint projected triangle on int image (CV_32I) by to single color */
    inline void paintTriangleSingleColor(
            cv::Mat& output_image,
            const cv::Size& image_size,
            const std::vector<cv::Point2f>& projected,
            const cv::Vec3i& face,
            int color) {
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
                    output_image.at<int32_t>(j, i) = color;
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
                    output_image.at<int32_t>(j, i) = color;
                }
            }
        }
    }
}

namespace ark {
    AvatarModel::AvatarModel(const std::string & model_dir) : MODEL_DIR(model_dir) {
        using namespace boost::filesystem;
        path modelPath = model_dir.empty() ? util::resolveRootPath("data/avatar-model") : model_dir;
        path skelPath = modelPath / "skeleton.txt";
        path jrPath = modelPath / "joint_regressor.txt";
        path jsrPath = modelPath / "joint_shape_regressor.txt";
        path posePriorPath = modelPath / "pose_prior.txt";
        path meshPath = modelPath / "mesh.txt";

        baseCloud = loadPCDToPointVectorFast((modelPath / "model.pcd").string());

        int nJoints, nPoints;
        // Read skeleton file
        std::ifstream skel(skelPath.string());
        if (!skel) {
            std::cerr << "ERROR: Avatar model is invalid, skeleton file not found\n";
            std::exit(0);
        }
        skel >> nJoints >> nPoints;

        // Assume joints are given in topologically sorted order
        parent.resize(nJoints);
        initialJointPos.resize(3, nJoints);
        for (int i = 0; i < nJoints; ++i) {
            int id;
            std::string _name; // throw away

            skel >> id;
            skel >> parent[id];
            skel >> _name >> initialJointPos(0, i)
                 >> initialJointPos(1, i) >> initialJointPos(2, i);
        }
        parent[0] = -1; // This should be in skeleton file, but just to make sure

        if (!skel) {
            std::cerr << "ERROR: Invalid avatar skeleton file: joint assignments are not present\n";
            std::exit(0);
        }

        // Process joint assignments
        assignedPoints.resize(nJoints);
        for (int i = 0; i < nJoints; ++i) {
            assignedPoints[i].reserve(7000 / nJoints);
        }
        size_t totalAssignments = 0;
        assignedJoints.resize(nPoints);
        for (int i = 0; i < nPoints; ++i) {
            int nEntries; skel >> nEntries;
            assignedJoints[i].reserve(nEntries);
            for (int j = 0; j < nEntries; ++j) {
                int joint; double w;
                skel >> joint >> w;
                assignedPoints[joint].emplace_back(w, i);
                assignedJoints[i].emplace_back(w, joint);
            }
            std::sort(assignedJoints[i].begin(), assignedJoints[i].end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                        return a.first > b.first;
                    });
            totalAssignments += nEntries;
        }

        size_t totalPoints = 0;
        assignStarts.resize(nJoints+1);
        assignWeights = Eigen::SparseMatrix<double>(totalAssignments, nPoints);
        assignWeights.reserve(Eigen::VectorXi::Constant(nPoints, 4));
        for (int i = 0; i < nJoints; ++i) {
            assignStarts[i] = totalPoints;
            for (auto& assignment : assignedPoints[i]) {
                int p = assignment.second;
                assignWeights.insert(totalPoints, p) = assignment.first;
                ++totalPoints;
            }
        }
        assignStarts[nJoints] = totalPoints;

        // Load all shape keys
        path keyPath = modelPath / "shapekey";
        if (is_directory(keyPath)) {
            int nShapeKeys = 0;
            for (directory_iterator it(keyPath); it != directory_iterator(); ++it) ++nShapeKeys;
            keyClouds.resize(3 * nPoints, nShapeKeys);

            int i = 0;
            for (directory_iterator it(keyPath); it != directory_iterator(); ++it) {
                keyClouds.col(i) = loadPCDToPointVectorFast(it->path().string());
                ++i;
            }
        } else {
            std::cerr << "WARNING: no shape key directory found for avatar\n";
        }

        // Load joint regressor / joint shape regressor
        std::ifstream jsr(jsrPath.string());
        if (jsr) {
            int nShapeKeys;
            jsr >> nShapeKeys;
            jointShapeRegBase.resize(nJoints * 3);
            jointShapeReg.resize(nJoints * 3, nShapeKeys);
            for (int i = 0; i < jointShapeRegBase.rows(); ++i) {
                jsr >> jointShapeRegBase(i);
            }
            for (int i = 0; i < jointShapeReg.rows(); ++i) {
                for (int j = 0; j < jointShapeReg.cols(); ++j) {
                    jsr >> jointShapeReg(i, j);
                }
            }
            useJointShapeRegressor = true;
            jsr.close();
        } else {
            std::ifstream jr(jrPath.string());
            jointRegressor = Eigen::SparseMatrix<double>(nPoints, nJoints);
            if (jr) {
                jr >> nJoints;
                jointRegressor.reserve(nJoints * 10);
                for (int i = 0; i < nJoints; ++i) {
                    int nEntries; jr >> nEntries;
                    int pointIdx; double val;
                    for (int j = 0; j < nEntries; ++j) {
                        jr >> pointIdx >> val;
                        jointRegressor.insert(pointIdx, i) = val;
                    }
                }
                jr.close();
            } else {
                std::cerr << "WARNING: joint regressor OR joint shape regressor found, model may be inaccurate with nonzero shapekey weights\n";
            }
            useJointShapeRegressor = false;
        }

        // Maybe load pose prior
        posePrior.load(posePriorPath.string());

        // Maybe load mesh
        std::ifstream meshFile(meshPath.string());
        if (meshFile) {
            int nFaces;
            meshFile >> nFaces;
            mesh.resize(3, nFaces);
            for (int i = 0; i < nFaces;++i) {
                meshFile >> mesh(0, i) >> mesh(1, i) >> mesh(2, i);
            }
        } else {
            std::cerr << "WARNING: mesh not found, maybe you are using an older version of avatar data files? "
                         "Some functions will not work.\n";
        }
    }

    Eigen::VectorXd AvatarPoseSequence::getFrame(size_t frame_id) const {
        if (preloaded) return data.col(frame_id);
        std::ifstream ifs(sequencePath, std::ios::in | std::ios::binary);
        ifs.seekg(frame_id * frameSize * sizeof(double), std::ios_base::beg);
        Eigen::VectorXd result(frameSize);
        ifs.read(reinterpret_cast<char*>(result.data()),
                 frameSize * sizeof(double));
        return result;
    }

    Avatar::Avatar(const AvatarModel& model) : model(model) {
        assignVecs.resize(3, model.assignWeights.rows());
        w.resize(model.numShapeKeys());
        r.resize(model.numJoints());
        w.setZero();
        p.setZero();
        for (int i = 0; i < model.numJoints(); ++i) {
            r[i].setIdentity();
        }
    }

    void Avatar::update() {
        // BEGIN_PROFILE;

        /** Apply shape keys */
        shapedCloudVec.noalias() = model.keyClouds * w + model.baseCloud; 
        Eigen::Map<CloudType> shapedCloud(shapedCloudVec.data(), 3, model.numPoints());

        /** Apply joint [shape] regressor */
        if (model.useJointShapeRegressor) {
            jointPos.resize(3, model.numJoints());
            Eigen::Map<Eigen::VectorXd> jointPosVec(jointPos.data(), 3 * model.numJoints());
            jointPosVec.noalias() = model.jointShapeRegBase + model.jointShapeReg * w;
        } else {
            jointPos.noalias() = shapedCloud * model.jointRegressor;
        }

        /** Update joint assignment/position constants */
        size_t j = 0;
        for (int i = 0; i < model.numJoints(); ++i) {
            auto col = jointPos.col(i);
            for (auto& assignment : model.assignedPoints[i]) {
                int idx = assignment.second;
                assignVecs.col(j++).noalias() = shapedCloud.col(idx) - col;
            }
        }

        for (int i = model.numJoints()-1; i > 0; --i) {
            jointPos.col(i).noalias() -= jointPos.col(model.parent[i]);
        }
        /** END of shape update, BEGIN pose update */

        /** Compute each joint's transform */
        //jointRot.clear();
        jointRot.resize(model.numJoints());
        jointRot[0].noalias() = r[0];

        jointPos.col(0) = p; /** Add root position to all joints */
        for (size_t i = 1; i < model.numJoints(); ++i) {
            jointRot[i].noalias() = jointRot[model.parent[i]] * r[i];
            jointPos.col(i) = jointRot[model.parent[i]] * jointPos.col(i) + jointPos.col(model.parent[i]);
        }

        /** Compute each point's transform */
        for (int i = 0; i < model.numJoints(); ++i) {
            Eigen::Map<CloudType> block(assignVecs.data() + 3 * model.assignStarts[i], 3, model.assignStarts[i+1] - model.assignStarts[i]);
            block = jointRot[i] * block;
            block.colwise() += jointPos.col(i);
        }
        cloud.noalias() = assignVecs * model.assignWeights;
        // PROFILE(UPDATE New);
    }

    void Avatar::randomize(bool randomize_pose,
        bool randomize_shape, bool randomize_root_pos_rot, uint32_t seed) {
        thread_local static std::mt19937 rg(std::random_device{}());
        if (~seed) {
            rg.seed(seed);
        }

        // Shape keys
        if (randomize_shape) {
            for (int i = 0; i < model.numShapeKeys(); ++i) {
                w(i) = random_util::randn(rg);
            }
        }

        // Pose
        if (randomize_pose) {
            auto samp = model.posePrior.sample();
            for (int i = 0; i < model.numJoints()-1; ++i) {
                // Axis-angle to rotation matrix
                Eigen::AngleAxisd angleAxis;
                angleAxis.angle() = samp.segment<3>(i*3).norm();
                angleAxis.axis() = samp.segment<3>(i*3) / angleAxis.angle();
                r[i + 1] = angleAxis.toRotationMatrix();
            }
        }

        if (randomize_root_pos_rot) {
            // Root position
            Eigen::Vector3d pos;
            pos.x() = random_util::uniform(rg, -1.0, 1.0);
            pos.y() = random_util::uniform(rg, -0.5, 0.5);
            pos.z() = random_util::uniform(rg, 2.2, 4.5);
            p = pos;

            // Root rotation
            const Eigen::Vector3d axis_up(0., 1., 0.);
            double angle_up  = random_util::uniform(rg, -M_PI / 3., M_PI / 3.) + M_PI;
            Eigen::AngleAxisd aa_up(angle_up, axis_up);

            double theta = random_util::uniform(rg, 0, 2 * M_PI);
            double phi   = random_util::uniform(rg, -M_PI/2, M_PI/2);
            Eigen::Vector3d axis_perturb;
            fromSpherical(1.0, theta, phi, axis_perturb);
            double angle_perturb = random_util::randn(rg, 0.0, 0.2);
            Eigen::AngleAxisd aa_perturb(angle_perturb, axis_perturb);

            r[0] = (aa_perturb * aa_up).toRotationMatrix();
        }
    }

    void randomMocapPose() {
        random_util::randint(0, 100);
    }

    Eigen::VectorXd Avatar::smplParams() const {
        Eigen::VectorXd res;
        res.resize((model.numJoints() - 1) * 3);
        for (int i = 1; i < model.numJoints(); ++i) {
            Eigen::AngleAxisd aa;
            aa.fromRotationMatrix(r[i]);
            res.segment<3>((i-1) * 3) = aa.axis() * aa.angle();
        }
        return res;
    }

    double Avatar::pdf() const {
        return model.posePrior.pdf(smplParams());
    }

    void Avatar::alignToJoints(const CloudType & pos)
    {
        ARK_ASSERT(pos.cols() == SmplJoint::_COUNT, "Joint number mismatch");

        Eigen::Vector3d vr = model.initialJointPos.col(SmplJoint::SPINE1) - model.initialJointPos.col(SmplJoint::ROOT_PELVIS);
        Eigen::Vector3d vrt = pos.col(SmplJoint::SPINE1) - pos.col(SmplJoint::ROOT_PELVIS);
        if (!std::isnan(pos(0, 0))) {
            p = pos.col(0);
        }
        if (!std::isnan(vr.x()) && !std::isnan(vrt.x())) {
            r[0] = Eigen::Quaterniond::FromTwoVectors(vr, vrt).toRotationMatrix();
        } else{
            r[0].setIdentity();
        }

        std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > rotTrans(pos.cols());
        rotTrans[0] = r[0];

        double scaleAvg = 0.0;
        for (int i = 1; i < pos.cols(); ++i) {
            scaleAvg += (pos.col(i) - pos.col(model.parent[i])).norm() /
                (model.initialJointPos.col(i) - model.initialJointPos.col(model.parent[i])).norm();
        }
        scaleAvg /= (pos.cols() - 1.0);
        double baseScale = (model.initialJointPos.col(SmplJoint::SPINE2) - model.initialJointPos.col(SmplJoint::ROOT_PELVIS)).norm() * (scaleAvg - 1.0);

        /** units to increase shape key 0 by to widen the avatar by approximately 1 meter */
        const double PC1_DIST_FACT = 32.0;
        w[0] = baseScale * PC1_DIST_FACT;
        if (std::isnan(w[0])) w[0] = 1.5;

        for (int i = 1; i < pos.cols(); ++i) {
            rotTrans[i] = rotTrans[model.parent[i]];
            if (!std::isnan(pos(0, i))) {
                Eigen::Vector3d vv = model.initialJointPos.col(i) - model.initialJointPos.col(model.parent[i]);
                Eigen::Vector3d vvt = pos.col(i) - pos.col(model.parent[i]);
                rotTrans[i] = Eigen::Quaterniond::FromTwoVectors(vv, vvt).toRotationMatrix();
                r[i] = rotTrans[model.parent[i]].transpose() * rotTrans[i];
            } else {
                r[i].setIdentity();
            }
        }
    }

    namespace avatar_pcl {
        pcl::PointCloud<pcl::PointXYZ>::Ptr getCloud(const Avatar& ava) {
            auto pointCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
            pointCloud->points.resize(ava.model.numPoints());
            pointCloud->getMatrixXfMap().block<3, Eigen::Dynamic>(0, 0, 3, pointCloud->size()) = ava.cloud.cast<float>();
            return pointCloud;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr getJointCloud(const Avatar& ava) {
            auto pointCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
            pointCloud->resize(ava.model.numJoints());
            pointCloud->getMatrixXfMap().block<3, Eigen::Dynamic>(0, 0, 3, pointCloud->size()) = ava.jointPos.cast<float>();
            return pointCloud;
        }

        pcl::PolygonMesh::Ptr getMesh(const Avatar& ava) {
            auto pclCloud = getCloud(ava);
            auto mesh = boost::make_shared<pcl::PolygonMesh>();
            pcl::toPCLPointCloud2(*pclCloud, mesh->cloud);

            mesh->polygons.reserve(ava.model.numFaces());
            for (int i = 0; i < ava.model.numFaces();++i) {
                mesh->polygons.emplace_back();
                auto& face = mesh->polygons.back().vertices; 
                face.resize(3);
                for (int j = 0; j < 3; ++j)
                    face[j] = ava.model.mesh(j, i);
            }
            return mesh;
        }
    }


    AvatarRenderer::AvatarRenderer(const Avatar& ava, const CameraIntrin& intrin)
        : ava(ava), intrin(intrin) { }

    const std::vector<cv::Point2f>& AvatarRenderer::getProjectedPoints() const {
        if (projectedPoints.empty()) {
            projectedPoints.resize(ava.model.numPoints());
            for (size_t i = 0; i < ava.cloud.cols(); ++i) {
                const auto& pt = ava.cloud.col(i);
                projectedPoints[i].x = static_cast<double>(pt(0))
                    * intrin.fx / pt(2) + intrin.cx;
                projectedPoints[i].y = -static_cast<double>(pt(1)) * intrin.fy / pt(2) + intrin.cy;
            } 
        }
        return projectedPoints;
    }

    const std::vector<cv::Point2f>& AvatarRenderer::getProjectedJoints() const {
        if (projectedJoints.empty()) {
            projectedJoints.resize(ava.model.numJoints());
            for (size_t i = 0; i < ava.jointPos.cols(); ++i) {
                const auto& pt = ava.jointPos.col(i);
                projectedJoints[i].x = static_cast<double>(pt(0))
                    * intrin.fx / pt(2) + intrin.cx;
                projectedJoints[i].y = -static_cast<double>(pt(1)) * intrin.fy / pt(2) + intrin.cy;
            } 
        }
        return projectedJoints;
    }

    const std::vector<AvatarRenderer::FaceType>& AvatarRenderer::getOrderedFaces() const {
        if (orderedFaces.empty()) {
            static auto faceComp = [](const FaceType& a, const FaceType& b) {
                return a.first > b.first;
            };

            orderedFaces.reserve(ava.model.numFaces());
            for (int i = 0; i < ava.model.numFaces();++i) {
                const auto& face = ava.model.mesh.col(i);
                orderedFaces.emplace_back(0.f, cv::Vec3i(face(0), face(1), face(2)));
            }

            // Sort faces by decreasing center depth
            // so that when painted front faces will cover back faces
            for (int i = 0; i < ava.model.numFaces();++i) {
                auto& face = orderedFaces[i].second;
                orderedFaces[i].first =
                    (ava.cloud(2, face[0]) + ava.cloud(2, face[1]) + ava.cloud(2, face[2])) / 3.f;
            }
            std::sort(orderedFaces.begin(), orderedFaces.end(), faceComp);
        }
        return orderedFaces;
    }

    cv::Mat AvatarRenderer::renderDepth(const cv::Size& image_size) const {
        const auto& projected = getProjectedPoints();
        const auto& faces = getOrderedFaces();

        cv::Mat renderedDepth = cv::Mat::zeros(image_size, CV_32F);
        for (int i = 0; i < ava.model.numFaces();++i) {
            paintDepthTriangleBary(renderedDepth, image_size, projected, ava.cloud, faces[i].second);
        }
        return cv::max(renderedDepth, 0.0f);
    }

    cv::Mat AvatarRenderer::renderPartMask(const cv::Size& image_size, const int* part_map) const {
        const auto& projected = getProjectedPoints();
        const auto& faces = getOrderedFaces();

        cv::Mat partMaskMap = cv::Mat::zeros(image_size, CV_8U);
        partMaskMap.setTo(255);
        for (int i = 0; i < ava.model.numFaces(); ++i) {
            paintPartsTriangleNN(partMaskMap, image_size, projected, ava.model.assignedJoints, faces[i].second, part_map);
        }
        return partMaskMap;
    }

    cv::Mat AvatarRenderer::renderFaces(const cv::Size& image_size) const {
        const auto& projected = getProjectedPoints();
        const auto& faces = getOrderedFaces();

        cv::Mat facesMap = cv::Mat::zeros(image_size, CV_32S);
        facesMap.setTo(-1);
        for (int i = 0; i < ava.model.numFaces(); ++i) {
            paintTriangleSingleColor(facesMap, image_size, projected, faces[i].second, i);
        }
        return facesMap;
    }

    void AvatarRenderer::update() const {
        projectedPoints.clear();
        projectedJoints.clear();
        orderedFaces.clear();
    }

    AvatarPoseSequence::AvatarPoseSequence(
            const std::string& pose_sequence_path) {
        using namespace boost::filesystem;
        path seqPath = pose_sequence_path.empty() ? util::resolveRootPath("data/avatar-mocap/cmu-mocap.dat") : pose_sequence_path;
        path metaPath(std::string(seqPath.string()).append(".txt"));

        if (!exists(seqPath) || !exists(metaPath)) {
            numFrames = 0;
            return;
        }
        sequencePath = seqPath.string();
        
        std::ifstream metaIfs(metaPath.string());
        size_t nSubseq, frameSizeBytes, subseqStart;
        metaIfs >> nSubseq >> numFrames >> frameSizeBytes;
        std::string subseqName;
        for (int sid = 0; sid < nSubseq; ++sid) {
            metaIfs >> subseqStart >> subseqName;
            subsequences[subseqName] = subseqStart / frameSizeBytes;
        }
        metaIfs.close();

        frameSize = frameSizeBytes / sizeof(double);
    }

    void AvatarPoseSequence::poseAvatar(Avatar& ava, size_t frame_id) const {
        if (preloaded) {
            auto frameData = data.col(frame_id);
            ava.p = frameData.head<3>();
            Eigen::Quaterniond q;
            for (int i = 0; i < ava.r.size(); ++i) { 
                q.coeffs().noalias() = frameData.segment<4>(i * 4 + 3);
                ava.r[i].noalias() = q.toRotationMatrix();
            }
        } else{
            Eigen::VectorXd frameData = getFrame(frame_id);
            ava.p = frameData.head<3>();
            Eigen::Quaterniond q;
            for (int i = 0; i < ava.r.size(); ++i) { 
                q.coeffs().noalias() = frameData.segment<4>(i * 4 + 3);
                ava.r[i].noalias() = q.toRotationMatrix();
            }
        }
    }

    void AvatarPoseSequence::preload() {
        data.resize(frameSize, numFrames);
        std::ifstream ifs(sequencePath, std::ios::in | std::ios::binary);
        ifs.read(reinterpret_cast<char*>(data.data()),
                 numFrames * frameSize * sizeof(double));
        preloaded = true;
    }
}
