#include "Avatar.h"

#include <fstream>
#include <boost/filesystem.hpp>

#include "Version.h"
#include "Util.h"

namespace ark {
Eigen::VectorXd AvatarPoseSequence::getFrame(size_t frame_id) const {
    if (preloaded) return data.col(frame_id);
    std::ifstream ifs(sequencePath, std::ios::in | std::ios::binary);
    ifs.seekg(frame_id * frameSize * sizeof(double), std::ios_base::beg);
    Eigen::VectorXd result(frameSize);
    ifs.read(reinterpret_cast<char*>(result.data()),
             frameSize * sizeof(double));
    return result;
}

AvatarPoseSequence::AvatarPoseSequence(const std::string& pose_sequence_path) {
    using namespace boost::filesystem;
    path seqPath =
        pose_sequence_path.empty()
            ? util::resolveRootPath("data/avatar-mocap/cmu-mocap.dat")
            : pose_sequence_path;
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
        ava.p.noalias() = frameData.head<3>();
        Eigen::Quaterniond q;
        for (int i = 0; i < ava.r.size(); ++i) {
            q.coeffs().noalias() = frameData.segment<4>(i * 4 + 3);
            ava.r[i].noalias() = q.toRotationMatrix();
        }
    } else {
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
}  // namespace ark
