#include "ViconSkeleton.h"

#include <iostream>
#include <fstream>
#include "Avatar.h"
#include "Util.h"

namespace {
    inline Eigen::Quaterniond eulerToQuat(double rx, double ry, double rz) {
        static constexpr double D2R = M_PI / 180.0;
        return Eigen::AngleAxisd(rz * D2R, Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(ry * D2R, Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(rx * D2R, Eigen::Vector3d::UnitX());
    }
}  // namespace

namespace ark {
    ViconSkeleton::ViconSkeleton() : data((int)JointType::_COUNT * 3) {
        joints.reserve((int)JointType::_COUNT);
        for (int i = 0; i < (int)JointType::_COUNT; ++i) {
            joints.emplace_back(i, &data[0] + i * 3);
        }
        std::fill(data.begin(), data.end(), 0.0);
    }

    ViconSkeleton::ViconSkeleton(const std::string & asf, const std::string & amc,
        int frame, float length_scale) : data((int)JointType::_COUNT * 3) {
        joints.reserve((int)JointType::_COUNT);
        for (int i = 0; i < (int)JointType::_COUNT; ++i) {
            joints.emplace_back(i, &data[0] + i * 3);
        }
        loadASF(asf, false, length_scale);
        loadAMC(amc, frame);
    }

    void ViconSkeleton::loadASF(const std::string & asf_file, bool load_rest_pose,
        float length_scale)
    {
        unload();

        std::ifstream asf(asf_file);
        if (!asf) {
            std::cerr << "WARNING: Failed to open ASF file: " << asf_file << "\n";
            return;
        }

        // initialize
        dof.resize(joints.size());
        axes.resize(joints.size(), Eigen::Quaterniond::Identity());
        axesInv.resize(joints.size(), Eigen::Quaterniond::Identity());
        refPos.resize(joints.size(), Eigen::Vector3d(0,0,0));

        // ASF section: 0=none 1=units 2=root 3=bonedata 4=hierarchy
        int section = 0, boneID = 0;
        Joint * jnt;
        std::string line;

        for (int lineNum = 1; asf; ++lineNum) {
            std::getline(asf, line);
            util::trim(line);
            if (line.empty() || line[0] == '#') continue;
            if (line[0] == ':') {
                // section change
                if (line == ":units") section = 1;
                else if (line == ":root") section = 2;
                else if (line == ":bonedata") section = 3;
                else if (line == ":hierarchy") section = 4;
                else section = 0;
                continue;
            }
            else if (section == 0) continue;

            // normal line
            std::vector<std::string> spl = util::split(line, " ", true, true);
            if (spl.empty()) continue;

            switch (section) {
            case 1:
                if (spl[0] == "length" && spl.size() > 1) 
                    length_scale /= std::atof(spl[1].c_str());
                break;
            case 2:
                if (spl[0] == "position" && spl.size() > 3) {
                    refPos[0].x() = std::atof(spl[1].c_str());
                    refPos[0].y() = std::atof(spl[2].c_str());
                    refPos[0].z() = std::atof(spl[3].c_str());
                } else if (spl[0] == "order") {
                    dof[0].clear();
                    for (size_t k = 1; k < spl.size(); ++k) {
                        util::lower(spl[k]);
                        dof[0].push_back(spl[k]);
                    }
                }
                break;
            case 3:

                if (line == "end") boneID = 0; // invalidate ID
                else {
                    if (spl[0] == "id") {
                        boneID = std::atoi(spl[1].c_str());
                        if (boneID > 30) {
                            std::cerr << "ERROR: ASF file parser currently only supports up to 30 bones.\n";
                            return;
                        }
                        jnt = &joints[boneID];
                        jnt->index = boneID;
                    } else if (boneID > 0) {
                        if (spl[0] == "name") {
                            // assumes id comes before
                            if (spl.size() < 2) continue;
                            jnt->name = spl[1];
                            jointByName[jnt->name] = boneID;
                            util::upper(jnt->name);
                        } else if (spl[0] == "direction") {
                            if (spl.size() < 4) {
                                std::cerr << "WARNING: ASF syntax error ('" << asf_file << "': " << lineNum << ")\n";
                                continue;
                            }
                            refPos[boneID].x()= std::atof(spl[1].c_str());
                            refPos[boneID].y() = std::atof(spl[2].c_str());
                            refPos[boneID].z() = std::atof(spl[3].c_str());
                        } else if (spl[0] == "length") {
                            // note: assumes that length comes after direction!
                            if (spl.size() < 2) {
                                std::cerr << "WARNING: ASF syntax error ('" << asf_file << "': " << lineNum << ")\n";
                                continue;
                            }

                            double norm = refPos[boneID].norm();
                            if (norm != 0.0) {
                                double scale = std::atof(spl[1].c_str()) * length_scale;
                                refPos[boneID] *= scale / refPos[boneID].norm();
                            }
                        }
                        else if (spl[0] == "axis") {
                            if (spl.size() < 4) {
                                std::cerr << "WARNING: ASF syntax error ('" << asf_file << "': " << lineNum << ")\n";
                                continue;
                            }
                            // only degrees supported
                            double x = std::atof(spl[1].c_str()); 
                            double y = std::atof(spl[2].c_str()); 
                            double z = std::atof(spl[3].c_str());
                            axes[boneID] = eulerToQuat(x, y, z);
                            axesInv[boneID] = axes[boneID].inverse();
                        }
                        else if (spl[0] == "dof") {
                            dof[boneID].clear();
                            for (size_t k = 1; k < spl.size(); ++k) {
                                util::lower(spl[k]);
                                dof[boneID].push_back(spl[k]);
                            }
                        }
                    }
                    // else just continue to next line
                }
                break;

            case 4:
                // build hierarchy
                if (line.length() <= 1) continue;
                boneID = jointByName[spl[0]];
                for (size_t k = 1; k < spl.size(); ++k) {
                    int childID = jointByName[spl[k]];
                    joints[boneID].child.push_back(&joints[childID]);
                    joints[childID].parent = &joints[boneID];
                }
            }
        }

        asf.close();
        lengthScale = length_scale;

        // enter data for rest pose (AMC 'frame 0')
        amcData.resize(1);
        amcData[0].resize(joints.size() * 3, 0.0);
        std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > tmp;
        initAMCFrame(amcData[0], tmp);
        asfLoaded = true;
        if (load_rest_pose) {
            loadFrame(0);
        }
    }

    void ViconSkeleton::loadAMC(const std::string & amc_file, int frame)
    {
        amcData.resize(1);
        amcLoaded = false;
        if (!asfLoaded) {
            std::cerr << "ERROR: Cannot load an AMC file before an ASF file.\n";
            return;
        }

        std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > rot(joints.size());

        // load AMC
        std::ifstream amc(amc_file);
        if (!amc) {
            std::cerr << "WARNING: Failed to open AMC file: " << amc_file << "\n";
        }

        int frmID = -1;
        std::string line;

        for (int lineNum = 1; amc; ++lineNum) {
            std::getline(amc, line);
            util::trim(line);

            if (line.empty() || line[0] == '#' || line[0] == ':') continue;

            std::vector<std::string> spl = util::split(line, " ", true, true);
            if (spl.size() == 1 && spl[0][0] >= '0' && spl[0][0] <= '9') {
                if (frmID > 0) initAMCFrame(amcData[frmID], rot);
                std::fill(rot.begin(), rot.end(), Eigen::Quaterniond::Identity());
                frmID = std::atoi(spl[0].c_str());
                while ((int)amcData.size() <= frmID) amcData.emplace_back();
                amcData[frmID].resize(joints.size() * 3, 0.0);
            }
            else if (frmID > 0) {
                int boneID = jointByName[spl[0]];
                double rx = 0.0, ry = 0.0, rz = 0.0;
                for (size_t k = 1; k < spl.size(); ++k) {
                    double val = std::atof(spl[k].c_str());
                    std::string & dim = dof[boneID][k - 1];
                    if (dim[0] == 't') {
                        val *= lengthScale;
                        if (dim[1] == 'x') amcData[frmID][boneID * 3] = val;
                        else if (dim[1] == 'y') amcData[frmID][boneID * 3 + 1] = val;
                        else if (dim[1] == 'z') amcData[frmID][boneID * 3 + 2] = val;
                    }
                    else if (dim[0] == 'r') {
                        if (dim[1] == 'x') rx = val;
                        else if (dim[1] == 'y') ry = val;
                        else if (dim[1] == 'z') rz = val;
                    }
                }
                rot[boneID] = axes[boneID] * eulerToQuat(rx, ry, rz) * axesInv[boneID];
            }
        }
        if (frmID > 0) initAMCFrame(amcData[frmID], rot);
        amc.close();

        amcLoaded = true;
        if (frame >= amcData.size()) {
            std::cerr <<
                "WARNING: requested frame " << frame << " is not present in the AMC file.\n";
            rest();
        } else {
            loadFrame(frame);
        }
    }

    void ViconSkeleton::loadFrame(int frame)
    {
        if (!amcLoaded && (!asfLoaded || frame)) {
            std::cerr <<
                "WARNING: cannot call loadFrame, rest, nextFrame, etc. before an AMC file is loaded.\n";
            return;
        } else if (frame < 0 || frame >= amcData.size()) {
            std::cerr <<
                "WARNING: frame number " << frame << " out of bounds.\n";
            return;
        }
        std::copy(amcData[frame].begin(), amcData[frame].end(), data.begin());
        curFrame = frame;
    }

    void ViconSkeleton::rest()
    {
        loadFrame(0);
    }

    void ViconSkeleton::unload()
    {
        asfLoaded = amcLoaded = false;
        jointByName.clear(); amcData.clear();
        dof.clear(); axes.clear(); axesInv.clear();
        refPos.clear();
        lengthScale = curFrame = 0;
        std::fill(data.begin(), data.end(), 0.0);
    }


    bool ViconSkeleton::nextFrame(int num, bool loop)
    {
        if (!amcLoaded || curFrame == 0) return false;
        int frmID = curFrame + num;
        if (loop) {
            if (frmID <= 0) frmID += (int)amcData.size() - 1;
            if (frmID >= (int) amcData.size()) frmID -= (int)amcData.size() - 1;
        }
        else if (frmID <= 0 || frmID >= amcData.size()) return false;
        loadFrame(frmID);
        return true;
    }

    bool ViconSkeleton::prevFrame(int num, bool loop)
    {
        return nextFrame(-num, loop);
    }

    bool ViconSkeleton::isASFLoaded() const
    {
        return asfLoaded;
    }

    bool ViconSkeleton::isAMCLoaded() const
    {
        return amcLoaded;
    }

    int ViconSkeleton::frame() const
    {
        return curFrame;
    }

    int ViconSkeleton::numFrames() const
    {
        if (!amcLoaded) return 0;
        return (int) amcData.size();
    }

    ViconSkeleton::Joint & ViconSkeleton::getJointByName(const std::string & name)
    {
        return joints[jointByName[name]];
    }

    /** Align avatar to skeleton in current frame */
    Eigen::Matrix<double, 3, Eigen::Dynamic> ViconSkeleton::getSmplJoints() const {
        // stores SMPL joint positions
        Eigen::Matrix<double, 3, Eigen::Dynamic> p(3, (int) SmplJoint::_COUNT);

        // 'forward'-facing direction for avatar
        Eigen::Vector3d forward = joints[(int)ViconSkeleton::JointType::SPINE2].pos - joints[(int)ViconSkeleton::JointType::HIP_C].pos;
        forward = forward.cross(joints[(int)ViconSkeleton::JointType::HIP_R].pos - joints[(int)ViconSkeleton::JointType::HIP_L].pos);
        forward.normalize();

        // translate ViconSkeleton joint positions -> HumanAvatar (SMPL) joint positions
        p.col(SmplJoint::ROOT_PELVIS) = joints[(int)ViconSkeleton::JointType::HIP_C].pos;

        p.col(SmplJoint::R_HIP) = joints[(int)ViconSkeleton::JointType::HIP_R].pos;
        p.col(SmplJoint::L_HIP) = joints[(int)ViconSkeleton::JointType::HIP_L].pos;
        p.col(SmplJoint::R_KNEE) = joints[(int)ViconSkeleton::JointType::KNEE_R].pos;
        p.col(SmplJoint::L_KNEE) = joints[(int)ViconSkeleton::JointType::KNEE_L].pos;
        p.col(SmplJoint::R_ANKLE) = joints[(int)ViconSkeleton::JointType::ANKLE_R].pos;
        p.col(SmplJoint::L_ANKLE) = joints[(int)ViconSkeleton::JointType::ANKLE_L].pos;
        p.col(SmplJoint::R_FOOT) = joints[(int)ViconSkeleton::JointType::FOOT_R].pos;
        p.col(SmplJoint::L_FOOT) = joints[(int)ViconSkeleton::JointType::FOOT_L].pos;

        p.col(SmplJoint::SPINE1) = joints[(int)ViconSkeleton::JointType::SPINE1].pos;
        p.col(SmplJoint::SPINE2) = joints[(int)ViconSkeleton::JointType::SPINE2].pos;
        p.col(SmplJoint::SPINE3) = 0.5f * (joints[(int)ViconSkeleton::JointType::SHOULDER_C].pos +  joints[(int)ViconSkeleton::JointType::SPINE2].pos);
        p.col(SmplJoint::R_SHOULDER) = joints[(int)ViconSkeleton::JointType::SHOULDER_R].pos;
        p.col(SmplJoint::L_SHOULDER) = joints[(int)ViconSkeleton::JointType::SHOULDER_L].pos;
        p.col(SmplJoint::R_COLLAR) = 0.5f * (joints[(int)ViconSkeleton::JointType::SHOULDER_R].pos + joints[(int)ViconSkeleton::JointType::SHOULDER_C].pos);
        p.col(SmplJoint::L_COLLAR) = 0.5f * (joints[(int)ViconSkeleton::JointType::SHOULDER_L].pos + joints[(int)ViconSkeleton::JointType::SHOULDER_C].pos);
        p.col(SmplJoint::NECK) = joints[(int)ViconSkeleton::JointType::NECK2].pos;
        p.col(SmplJoint::HEAD) = joints[(int)ViconSkeleton::JointType::HEAD].pos + forward * 0.03f;

        p.col(SmplJoint::R_ELBOW) = joints[(int)ViconSkeleton::JointType::RHUMERUS].pos;
        p.col(SmplJoint::L_ELBOW) = joints[(int)ViconSkeleton::JointType::LHUMERUS].pos;
        p.col(SmplJoint::R_WRIST) = 0.5f * (joints[(int)ViconSkeleton::JointType::WRIST2_R].pos + joints[(int)ViconSkeleton::JointType::WRIST1_R].pos);
        p.col(SmplJoint::L_WRIST) = 0.5f * (joints[(int)ViconSkeleton::JointType::WRIST2_L].pos + joints[(int)ViconSkeleton::JointType::WRIST1_L].pos);
        p.col(SmplJoint::R_HAND) = joints[(int)ViconSkeleton::JointType::FINGERS_R].pos;
        p.col(SmplJoint::L_HAND) = joints[(int)ViconSkeleton::JointType::FINGERS_L].pos;
        return p;
    }

    void ViconSkeleton::initAMCFrame(std::vector<double> & trans, std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > & rot)
    { 
        std::vector<int> stk;
        stk.push_back((int) JointType::ROOT);

        while (!stk.empty()) {
            int boneID = stk.back();
            stk.pop_back();
            Joint & jnt = joints[boneID];
            Joint * parent = jnt.parent;
            double & x = trans[boneID * 3], & y = trans[boneID * 3 + 1], & z = trans[boneID * 3 + 2];

            if (!rot.empty()) {
                if (parent) rot[boneID] = rot[parent->index] * rot[boneID];
                Eigen::Vector3d rotated = rot[boneID]._transformVector(refPos[boneID]);
                x += rotated.x(); y += rotated.y(); z += rotated.z();
            }
            else {
                x += refPos[boneID].x(); y += refPos[boneID].y(); z += refPos[boneID].z();
            }

            if (parent) {
                x += trans[parent->index * 3]; y += trans[parent->index * 3 + 1]; z += trans[parent->index * 3 + 2];
            }

            for (Joint * childJnt : jnt.child) {
                stk.push_back(childJnt->index);
            }
        }
    }
}

