#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <Eigen/src/Core/IO.h>

#include "Avatar.h"
namespace {
    void writePCD(const std::string& path, const Eigen::Ref<const ark::CloudType>& data, const std::vector<int>& indices) {
        std::ofstream ofs(path);
        ofs << "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
               "WIDTH "  << std::to_string(indices.size())  << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
               "POINTS " << std::to_string(indices.size())  << "\nDATA ascii\n";
        ofs << std::fixed << std::setprecision(18);
        for (int i : indices) {
            ofs << data(0, i) << " " << data(1, i) << " " << data(2, i) << "\n";
        }
    }

    std::vector<std::string> SMPL_JOINT_NAMES = {
        "PELVIS", "L_HIP", "R_HIP", "SPINE1", "L_KNEE", "R_KNEE", "SPINE2", "L_ANKLE",
        "R_ANKLE", "SPINE3", "L_FOOT", "R_FOOT", "NECK", "L_COLLAR", "R_COLLAR", "HEAD", "L_SHOULDER",
        "R_SHOULDER", "L_ELBOW", "R_ELBOW", "L_WRIST", "R_WRIST", "L_HAND", "R_HAND"
    };
}

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    std::string outputPath;
    double thresh;
    std::string rootName;
    std::vector<std::string> deleteJoints;

    po::options_description desc("Option arguments");
    po::options_description descPositional("OpenARK SMPL partial avatar model creator (c) Alex Yu 2019\nPosition arguments");
    po::options_description descCombined("");
    desc.add_options()
        ("help", "produce help message")
        ("names,n", "print joint names")
        (",t", po::value<double>(&thresh)->default_value(0.6), "Threshold on total remaining joint weight (after removing joints) for keeping a joint")
        (",r", po::value<std::string>(&rootName)->default_value("PELVIS"), "New root joint id, e.g. -r SPINE1")
        (",d", po::value<std::vector<std::string> >(&deleteJoints)->multitoken()->zero_tokens()->composing(), "Joint id to delete (can be specified multiple times, like -d L_HIP -d R_HIP)")
    ;

    descPositional.add_options()
        ("output_path", po::value<std::string>(&outputPath)->required(), "Where to write output model files")
        ;

    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;
    
    po::positional_options_description posopt;
    posopt.add("output_path", 1);

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

    if ( vm.count("names")  )
    {
        for (const auto& name : SMPL_JOINT_NAMES) {
            std::cout << name << " ";
        }
        std::cout << "\n";
        return 0;
    }

    try {
        po::notify(vm);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << descPositional << "\n" << desc << "\n";
        return 1;
    }

    using namespace boost::filesystem;
    path outPath(outputPath);
    path shapekeysPath = outPath / "shapekey";
    if (!boost::filesystem::exists(outPath)) {
        boost::filesystem::create_directories(outPath);
    }
    if (!boost::filesystem::exists(shapekeysPath)) {
        boost::filesystem::create_directories(shapekeysPath);
    }

    ark::AvatarModel model;
    ark::Avatar ava(model);
    ava.update();

    std::vector<bool> keepJoints(model.numJoints(), false);
    std::vector<int> jointIndices;
    std::vector<int> revJointIndices(model.numJoints(), -1);

    for (auto& c: rootName) c = std::toupper(c);
    int root = std::find(SMPL_JOINT_NAMES.begin(), SMPL_JOINT_NAMES.end(), rootName) - SMPL_JOINT_NAMES.begin();
    if ((size_t)root == SMPL_JOINT_NAMES.size()) {
        std::cerr << rootName << ": not a valid joint name (root)\n";
        return 0;
    }
    keepJoints[root] = true;
    jointIndices.push_back(root);
    revJointIndices[root] = 0;
    for (int i = root + 1; i < model.numJoints(); ++i) {
        keepJoints[i] = keepJoints[model.parent(i)];
    }

    for (std::string& delName : deleteJoints) {
        for (auto& c: delName) c = std::toupper(c);
        int delId = std::find(SMPL_JOINT_NAMES.begin(), SMPL_JOINT_NAMES.end(), delName) - SMPL_JOINT_NAMES.begin();
        if ((size_t)delId == SMPL_JOINT_NAMES.size()) {
            std::cerr << delName << ": not a valid joint name (delete)\n";
            return 0;
        }
        if (delId <= root) {
            std::cerr << delName << ": cannot delete root or parent of root\n";
            return 0;
        }
        std::vector<bool> delJoints(model.numJoints(), false);
        delJoints[delId] = true;
        keepJoints[delId] = false;
        for (int i = delId + 1; i < model.numJoints(); ++i) {
            delJoints[i] = delJoints[model.parent(i)];
            if (delJoints[i]) {
                keepJoints[i] = false;
            }
        }
    }

    for (int i = root + 1; i < model.numJoints(); ++i) {
        if (keepJoints[i]) {
            revJointIndices[i] = jointIndices.size();
            jointIndices.push_back(i);
        }
    }

    std::vector<int> pointIndices;
    std::vector<int> revPointIndices(model.numPoints(), -1);
    pointIndices.reserve(model.numPoints());
    std::vector<std::vector<std::pair<double, int> > > pointAssignments(model.numPoints());
    for (int i = 0; i < model.numPoints(); ++i) {
        double total = 0.0;
        for (auto& assign : model.assignedJoints[i]) {
            if (keepJoints[assign.second]) {
                pointAssignments[i].emplace_back(assign);
                total += assign.first;
            }
        }
        if (pointAssignments[i].size() > 0 && total > thresh) {
            revPointIndices[i] = pointIndices.size();
            pointIndices.push_back(i);
            for (auto& assign : pointAssignments[i]) {
                assign.first /= total;
            }
        }
    }

    // Write skeleton.txt
    std::ofstream skel((outPath / "skeleton.txt").string());
    skel << std::fixed << std::setprecision(18);
    skel << jointIndices.size() << " " << pointIndices.size() << "\n";
    skel << std::fixed << std::setprecision(18);
    for (size_t i = 0; i < jointIndices.size(); ++i) {
        skel << i << " ";
        if (model.parent[jointIndices[i]] != -1)
            skel << revJointIndices[model.parent[jointIndices[i]]] << " ";
        else
            skel << -1 << " ";
        skel << SMPL_JOINT_NAMES[jointIndices[i]] << " "
             << model.initialJointPos(0, jointIndices[i]) << " "
             << model.initialJointPos(1, jointIndices[i]) << " "
             << model.initialJointPos(2, jointIndices[i]) << "\n";
    }

    for (size_t i = 0; i < pointIndices.size(); ++i) {
        skel << pointAssignments[pointIndices[i]].size();
        for (auto & assign : pointAssignments[pointIndices[i]]) {
            skel << " " << revJointIndices[assign.second] << " " << assign.first;
        }
        skel << "\n";
    }
    skel.close();

    // Write model.pcd
    Eigen::Map<ark::CloudType> baseCloud(model.baseCloud.data(), 3, model.numPoints());
    writePCD((outPath / "model.pcd").string(), baseCloud, pointIndices);

    // Write shape key.pcds
    for (int i = 0; i < model.numShapeKeys(); ++i) {
        std::stringstream ss_key_id;
        ss_key_id << std::setw(3) << std::setfill('0') << std::to_string(i);
        Eigen::Map<ark::CloudType> keyCloud(model.keyClouds.data() + model.keyClouds.rows() * i, 3, model.numPoints());

        writePCD((shapekeysPath / ("shape" + ss_key_id.str() + ".pcd")).string(), keyCloud, pointIndices);
    }
 
    // Prepare and write joint shape regressor
    if (!model.useJointShapeRegressor) {
        Eigen::Map<ark::CloudType> P(model.baseCloud.data(), 3, model.baseCloud.rows() / 3);
        model.jointShapeRegBase.resize(3 * model.numJoints());
        Eigen::Map<ark::CloudType> PJ(model.jointShapeRegBase.data(), 3, model.numJoints());
        PJ = P * model.jointRegressor;
        
        model.jointShapeReg.resize(model.numJoints() * 3,  model.numShapeKeys());
        for (int i = 0 ; i < model.numShapeKeys(); ++i) {
            Eigen::Map<ark::CloudType> Si(model.keyClouds.data() + model.keyClouds.rows() * i, 3, model.keyClouds.rows() / 3);
            Eigen::Map<ark::CloudType> PJ(model.jointShapeRegBase.data(), 3, model.numJoints());
            ark::CloudType SiJ = Si * model.jointRegressor;
            Eigen::Map<Eigen::VectorXd> SiJvec(SiJ.data(), model.numJoints() * 3);
            model.jointShapeReg.col(i).noalias() = SiJvec;
        }
    }

    std::ofstream jsr((outPath / "joint_shape_regressor.txt").string());
    jsr << std::fixed << std::setprecision(18);
    jsr << model.numShapeKeys() << "\n";
    for (int i = root; i < model.numJoints(); ++i) {
        if (keepJoints[i]) {
            if (i > root) jsr << " ";
            jsr << model.jointShapeRegBase(i*3) << " "
                << model.jointShapeRegBase(i*3+1) << " "
                << model.jointShapeRegBase(i*3+2);
        }
    }
    jsr << "\n";
    for (int i = root; i < model.numJoints(); ++i) {
        if (keepJoints[i]) {
            for (int j = 0; j <3; ++j) {
                for (int k = 0; k < model.numShapeKeys(); ++k) {
                    if (k) jsr << " ";
                    jsr << model.jointShapeReg(i*3 + j, k);
                }
                jsr << "\n";
            }
        }
    }
    jsr.close();

    if (model.hasMesh()) {
        // Write mesh.txt
        std::ofstream meshFile((outPath / "mesh.txt").string());
        int count = 0;
        for (int i = 0; i < model.mesh.cols(); ++i) {
            count += (revPointIndices[model.mesh(0, i)] >= 0 &&
                revPointIndices[model.mesh(1, i)] >= 0 &&
                revPointIndices[model.mesh(2, i)] >= 0);
        }
        meshFile << count << "\n";
        for (int i = 0; i < model.mesh.cols(); ++i) {
            int i1 = revPointIndices[model.mesh(0, i)];
            int i2 = revPointIndices[model.mesh(1, i)];
            int i3 = revPointIndices[model.mesh(2, i)];
            if (i1 >= 0 && i2 >= 0 && i3 >= 0) {
                meshFile << i1 << " " << i2 << " " << i3 << "\n";
            }
        }
        meshFile.close();
    }

    if (model.hasPosePrior()) {
        // Write pose_prior.txt
        std::ofstream ppFile((outPath / "pose_prior.txt").string());
        ppFile << std::fixed << std::setprecision(18);
        ppFile << model.posePrior.numComponents() << " " << jointIndices.size() * 3 - 3 << "\n";
        // Weights
        for (int i = 0; i < model.posePrior.numComponents(); ++i) {
            if (i) ppFile << " ";
            ppFile << model.posePrior.weight(i);
        }
        ppFile << "\n";
        // Means
        for (int i = 0; i < model.posePrior.numComponents(); ++i) {
            for (int j : jointIndices) {
                if (j == root) continue;
                if (j != jointIndices[1]) ppFile << " ";
                ppFile << model.posePrior.mean(i, (j-1)*3) << " " <<
                          model.posePrior.mean(i, (j-1)*3 + 1) << " " <<
                          model.posePrior.mean(i, (j-1)*3 + 2);
            }
            ppFile << "\n";
        }
        // Covs
        for (int i = 0; i < model.posePrior.numComponents(); ++i) {
            for (int j : jointIndices) {
                if (j == root) continue;
                for (int off = 0; off < 3; ++off) {
                    for (int k : jointIndices) {
                        if (k == root) continue;
                        if (k != jointIndices[1]) ppFile << " ";
                        ppFile << model.posePrior.cov[i]((j-1)*3+off, (k-1)*3) << " " <<
                            model.posePrior.cov[i]((j-1)*3+off, (k-1)*3+1) << " " <<
                            model.posePrior.cov[i]((j-1)*3+off, (k-1)*3+2);
                    }
                    ppFile << "\n";
                }
            }
        }
        ppFile.close();
    }
    
    return 0;
}
