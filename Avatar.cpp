#include "Avatar.h"

#include <chrono>
#include <iostream>

#include "Version.h"
#include "Util.h"

#include "internal/AvatarHelpers.h"

namespace ark {
Avatar::Avatar(const AvatarModel& model) : model(model) {
    w.resize(model.numShapeKeys());
    r.resize(model.numJoints());
    w.setZero();
    p.setZero();
    for (int i = 0; i < model.numJoints(); ++i) {
        r[i].setIdentity();
    }
}

void Avatar::update() {
    _ARK_BEGIN_PROFILE;

    /** Apply shape keys */
    shapedCloudVec.noalias() = model.keyClouds * w + model.baseCloud;
    Eigen::Map<CloudType> shapedCloud(shapedCloudVec.data(), 3,
                                      model.numPoints());
    // _ARK_PROFILE(SHAPE);

    /** Apply joint [shape] regressor */
    if (model.useJointShapeRegressor) {
        jointPos = model.initialJointPos;
        Eigen::Map<Eigen::VectorXd> jointPosVec(jointPos.data(),
                                                3 * model.numJoints());
        jointPosVec.noalias() += model.jointShapeReg * w;
    } else {
        jointPos.noalias() = shapedCloud * model.jointRegressor;
    }

    /** END of shape update, BEGIN pose update */

    /** Compute each joint's transform */
    jointTrans.resize(jointTrans.RowsAtCompileTime, model.numJoints());
    using TransformMap = Eigen::Map<Eigen::Matrix<double, 3, 4>>;
    /** Root joint joints */
    TransformMap jt0(jointTrans.data());
    jt0.leftCols<3>().noalias() = r[0];
    jt0.rightCols<1>() = p;  // Root position at center (non-standard!)
    for (size_t i = 1; i < model.numJoints(); ++i) {
        TransformMap jti(jointTrans.col(i).data());
        jti.leftCols<3>().noalias() = r[i];
        jti.rightCols<1>().noalias() =
            jointPos.col(i) - jointPos.col(model.parent[i]);
        util::mulAffine<double, Eigen::ColMajor>(
            TransformMap(jointTrans.col(model.parent[i]).data()), jti);
    }

    for (int i = 0; i < model.numJoints(); ++i) {
        TransformMap jti(jointTrans.col(i).data());
        Eigen::Vector3d jPosInit = jointPos.col(i);
        jointPos.col(i).noalias() = jti.rightCols<1>();
        jti.rightCols<1>().noalias() -= jti.leftCols<3>() * jPosInit;
    }

    /** Compute each point's transform */
    cloud.resize(3, model.numPoints());

    Eigen::Matrix<double, 12, Eigen::Dynamic> pointTrans = jointTrans * model.weights;
    for (size_t i = 0; i < model.numPoints(); ++i) {
        TransformMap pti(pointTrans.col(i).data());
        cloud.col(i).noalias() = pti * shapedCloud.col(i).homogeneous();
    }
    _ARK_PROFILE(UPDATE);
}

void Avatar::randomize(bool randomize_pose, bool randomize_shape,
                       bool randomize_root_pos_rot, uint32_t seed) {
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
        for (int i = 0; i < model.numJoints() - 1; ++i) {
            // Axis-angle to rotation matrix
            Eigen::AngleAxisd angleAxis;
            angleAxis.angle() = samp.segment<3>(i * 3).norm();
            angleAxis.axis() = samp.segment<3>(i * 3) / angleAxis.angle();
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
        double angle_up =
            random_util::uniform(rg, -M_PI / 3., M_PI / 3.) + M_PI;
        Eigen::AngleAxisd aa_up(angle_up, axis_up);

        double theta = random_util::uniform(rg, 0, 2 * M_PI);
        double phi = random_util::uniform(rg, -M_PI / 2, M_PI / 2);
        Eigen::Vector3d axis_perturb;
        fromSpherical(1.0, theta, phi, axis_perturb);
        double angle_perturb = random_util::randn(rg, 0.0, 0.2);
        Eigen::AngleAxisd aa_perturb(angle_perturb, axis_perturb);

        r[0] = (aa_perturb * aa_up).toRotationMatrix();
    }
}

Eigen::VectorXd Avatar::smplParams() const {
    Eigen::VectorXd res;
    res.resize((model.numJoints() - 1) * 3);
    for (int i = 1; i < model.numJoints(); ++i) {
        Eigen::AngleAxisd aa;
        aa.fromRotationMatrix(r[i]);
        res.segment<3>((i - 1) * 3) = aa.axis() * aa.angle();
    }
    return res;
}

double Avatar::pdf() const { return model.posePrior.pdf(smplParams()); }

void Avatar::alignToJoints(const CloudType& pos) {
    _ARK_ASSERT_EQ(pos.cols(), SmplJoint::_COUNT);

    Eigen::Vector3d vr = model.initialJointPos.col(SmplJoint::SPINE1) -
                         model.initialJointPos.col(SmplJoint::ROOT_PELVIS);
    Eigen::Vector3d vrt =
        pos.col(SmplJoint::SPINE1) - pos.col(SmplJoint::ROOT_PELVIS);
    if (!std::isnan(pos(0, 0))) {
        p = pos.col(0);
    }
    if (!std::isnan(vr.x()) && !std::isnan(vrt.x())) {
        r[0] = Eigen::Quaterniond::FromTwoVectors(vr, vrt).toRotationMatrix();
    } else {
        r[0].setIdentity();
    }

    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
        rotTrans(pos.cols());
    rotTrans[0] = r[0];

    double scaleAvg = 0.0;
    for (int i = 1; i < pos.cols(); ++i) {
        scaleAvg += (pos.col(i) - pos.col(model.parent[i])).norm() /
                    (model.initialJointPos.col(i) -
                     model.initialJointPos.col(model.parent[i]))
                        .norm();
    }
    scaleAvg /= (pos.cols() - 1.0);
    double baseScale = (model.initialJointPos.col(SmplJoint::SPINE2) -
                        model.initialJointPos.col(SmplJoint::ROOT_PELVIS))
                           .norm() *
                       (scaleAvg - 1.0);

    /** units to increase shape key 0 by to widen the avatar by approximately 1
     * meter */
    const double PC1_DIST_FACT = 32.0;
    w[0] = baseScale * PC1_DIST_FACT;
    if (std::isnan(w[0])) w[0] = 1.5;

    for (int i = 1; i < pos.cols(); ++i) {
        rotTrans[i] = rotTrans[model.parent[i]];
        if (!std::isnan(pos(0, i))) {
            Eigen::Vector3d vv = model.initialJointPos.col(i) -
                                 model.initialJointPos.col(model.parent[i]);
            Eigen::Vector3d vvt = pos.col(i) - pos.col(model.parent[i]);
            rotTrans[i] =
                Eigen::Quaterniond::FromTwoVectors(vv, vvt).toRotationMatrix();
            r[i] = rotTrans[model.parent[i]].transpose() * rotTrans[i];
        } else {
            r[i].setIdentity();
        }
    }
}
}  // namespace ark
