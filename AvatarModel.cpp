#include "Avatar.h"
#include <string>
#include <boost/filesystem.hpp>
#include <cnpy.h>

#include "Version.h"
#include "Util.h"
#include "UtilCnpy.h"

#include "internal/AvatarHelpers.h"

namespace ark {
// SMPL model loading code
AvatarModel::AvatarModel(const std::string& model_dir,
                         bool limit_one_joint_per_point)
    : MODEL_DIR(model_dir) {
    using namespace boost::filesystem;
    path modelPath = model_dir.empty()
                         ? util::resolveRootPath("data/avatar-model")
                         : model_dir;
    path npzPath = modelPath / "model.npz";
    path posePriorPath = modelPath / "pose_prior.txt";
    if (exists(npzPath)) {
        // New (npz) format
        cnpy::npz_t npz = cnpy::npz_load(npzPath.string());
        size_t n_verts = npz["v_template"].shape[0];
        size_t n_joints = npz["kintree_table"].shape[1];
        size_t n_faces = npz["f"].shape[0];
        size_t n_shape_blends = npz["shapedirs"].shape[2];
        size_t n_blend_shapes = n_shape_blends;

        using util::assertShape;

        // Load kintree
        const auto& kttable_raw = npz.at("kintree_table");
        parent.resize(n_joints);
        parent.noalias() = util::loadUintMatrix(kttable_raw, 2, n_joints)
                               .template topRows<1>()
                               .cast<int>()
                               .transpose();
        _ARK_ASSERT_EQ(parent[0], -1);

        // Load base template
        const auto& verts_raw = npz.at("v_template");
        assertShape(verts_raw, {n_verts, 3});
        baseCloud.noalias() =
            util::loadFloatMatrix(verts_raw, 1, n_verts * 3).transpose();

        // Load triangle mesh
        const auto& faces_raw = npz.at("f");
        assertShape(faces_raw, {n_faces, 3});
        mesh =
            util::loadUintMatrix(faces_raw, n_faces, 3).transpose().cast<int>();

        // Load joint regressor
        const auto& jreg_raw = npz.at("J_regressor");
        assertShape(jreg_raw, {n_joints, n_verts});
        jointRegressor.resize(n_joints, n_verts);
        jointRegressor = util::loadFloatMatrix(jreg_raw, n_joints, n_verts)
                             .transpose()
                             .sparseView();
        jointRegressor.makeCompressed();

        // Load LBS weights
        const auto& wt_raw = npz.at("weights");
        assertShape(wt_raw, {n_verts, n_joints});
        weights.resize(n_joints, n_verts);
        weights = util::loadFloatMatrix(wt_raw, n_verts, n_joints)
                      .transpose()
                      .sparseView();
        weights.makeCompressed();

        // (Compatibility with existing code)
        assignedJoints.resize(n_verts);
        assignedPoints.resize(n_joints);
        for (int k = 0; k < weights.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(weights, k); it;
                 ++it) {
                int r = it.row(), c = it.col();
                double wt = it.value();
                if (wt > 1e-12) {
                    assignedJoints[c].push_back({wt, r});
                    assignedPoints[r].push_back({wt, c});
                }
            }
        }
        for (int i = 0; i < n_joints; ++i) {
            std::sort(assignedPoints[i].begin(), assignedPoints[i].end(),
                      std::greater<std::pair<double, int>>());
        }
        for (int i = 0; i < n_verts; ++i) {
            std::sort(assignedJoints[i].begin(), assignedJoints[i].end(),
                      std::greater<std::pair<double, int>>());
        }

        // Blend shapes
        keyClouds.resize(3 * n_verts, n_blend_shapes);
        // Load shape-dep blend shapes
        const auto& sb_raw = npz.at("shapedirs");
        assertShape(sb_raw, {n_verts, 3, n_shape_blends});
        keyClouds.leftCols(n_shape_blends).noalias() =
            util::loadFloatMatrix(sb_raw, 3 * n_verts, n_shape_blends);

        // Load pose-dep blend shapes
        // (currently not used for efficiency reasons)
        // const auto& pb_raw = npz.at("posedirs");
        // assertShape(pb_raw, {n_verts, 3, n_pose_blends});
        // keyClouds.template rightCols<n_pose_blends>().noalias() =
        //     util::loadFloatMatrix(pb_raw, 3 * n_verts, n_pose_blends);

        // Compute the joint shape regressor
        useJointShapeRegressor = true;
        initialJointPos.noalias() =
            Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>>(
                baseCloud.data(), 3, n_verts) *
            jointRegressor;
        jointShapeRegBase.noalias() = Eigen::template Map<Eigen::VectorXd>(
            initialJointPos.data(),
            initialJointPos.rows() * initialJointPos.cols(), 1);
        jointShapeReg.resize(3 * n_joints, n_shape_blends);
        for (int i = 0; i < n_shape_blends; ++i) {
            Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>> keyVertsIn(
                keyClouds.col(i).data(), 3, n_verts);
            Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>> jointsOut(
                jointShapeReg.col(i).data(), 3, n_joints);
            jointsOut.noalias() = keyVertsIn * jointRegressor;
        }
    } else {
        std::cerr << "WARNING: Using deprecated ad-hoc SMPL model format; "
                     "please download SMPL .npz model and place it at "
                     "data/avatar-model/model.npz\n";
        // Old ad-hoc format
        path skelPath = modelPath / "skeleton.txt";
        path jrPath = modelPath / "joint_regressor.txt";
        path jsrPath = modelPath / "joint_shape_regressor.txt";
        path meshPath = modelPath / "mesh.txt";

        baseCloud =
            loadPCDToPointVectorFast((modelPath / "model.pcd").string());

        int nJoints, nPoints;
        // Read skeleton file
        std::ifstream skel(skelPath.string());
        if (!skel) {
            std::cerr
                << "ERROR: Avatar model is invalid, skeleton file not found\n";
            std::exit(0);
        }
        skel >> nJoints >> nPoints;

        // Assume joints are given in topologically sorted order
        parent.resize(nJoints);
        initialJointPos.resize(3, nJoints);
        for (int i = 0; i < nJoints; ++i) {
            int id;
            std::string _name;  // throw away

            skel >> id;
            skel >> parent[id];
            skel >> _name >> initialJointPos(0, i) >> initialJointPos(1, i) >>
                initialJointPos(2, i);
        }
        parent[0] =
            -1;  // This should be in skeleton file, but just to make sure

        if (!skel) {
            std::cerr
                << "ERROR: Invalid avatar skeleton file: joint assignments "
                   "are not present\n";
            std::exit(0);
        }

        // Process joint assignments
        weights.resize(nJoints, nPoints);
        weights.reserve(3 * nPoints);
        assignedPoints.resize(nJoints);
        for (int i = 0; i < nJoints; ++i) {
            assignedPoints[i].reserve(7000 / nJoints);
        }
        assignedJoints.resize(nPoints);
        for (int i = 0; i < nPoints; ++i) {
            int nEntries;
            skel >> nEntries;
            assignedJoints[i].reserve(nEntries);
            for (int j = 0; j < nEntries; ++j) {
                int joint;
                double w;
                skel >> joint >> w;
                assignedJoints[i].emplace_back(w, joint);
                weights.insert(joint, i) = w;
            }
            std::sort(assignedJoints[i].begin(), assignedJoints[i].end(),
                      [](const std::pair<double, int>& a,
                         const std::pair<double, int>& b) {
                          return a.first > b.first;
                      });
            if (limit_one_joint_per_point) {
                assignedJoints[i].resize(1);
                assignedJoints[i].shrink_to_fit();
                assignedJoints[i][0].first = 1.0;
                assignedPoints[assignedJoints[i][0].second].emplace_back(1.0,
                                                                         i);
            } else {
                for (int j = 0; j < nEntries; ++j) {
                    assignedPoints[assignedJoints[i][j].second].emplace_back(
                        assignedJoints[i][j].first, i);
                }
            }
        }

        // Load all shape keys
        path keyPath = modelPath / "shapekey";
        if (is_directory(keyPath)) {
            int nShapeKeys = 0;
            for (directory_iterator it(keyPath); it != directory_iterator();
                 ++it)
                ++nShapeKeys;
            keyClouds.resize(3 * nPoints, nShapeKeys);

            int i = 0;
            for (directory_iterator it(keyPath); it != directory_iterator();
                 ++it) {
                keyClouds.col(i) =
                    loadPCDToPointVectorFast(it->path().string());
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
                    int nEntries;
                    jr >> nEntries;
                    int pointIdx;
                    double val;
                    for (int j = 0; j < nEntries; ++j) {
                        jr >> pointIdx >> val;
                        jointRegressor.insert(pointIdx, i) = val;
                    }
                }
                jr.close();
            } else {
                std::cerr << "WARNING: neither joint regressor nor joint shape "
                             "regressor found, model may be inaccurate with "
                             "nonzero shapekey weights\n";
            }
            useJointShapeRegressor = false;
        }

        // Maybe load mesh
        std::ifstream meshFile(meshPath.string());
        if (meshFile) {
            int nFaces;
            meshFile >> nFaces;
            mesh.resize(3, nFaces);
            for (int i = 0; i < nFaces; ++i) {
                meshFile >> mesh(0, i) >> mesh(1, i) >> mesh(2, i);
            }
        } else {
            std::cerr
                << "WARNING: mesh not found, maybe you are using an older "
                   "version of avatar data files? "
                   "Some functions will not work.\n";
        }
    }

    size_t totalAssignments = 0;
    for (size_t i = 0; i < assignedJoints.size(); ++i) {
        totalAssignments += assignedJoints[i].size();
    }

    // Maybe load pose prior
    posePrior.load(posePriorPath.string());
}
}  // namespace ark
