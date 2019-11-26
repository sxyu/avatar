/** Configuration for smplsynth/partlabels */
#pragma once
#include "Avatar.h"

namespace ark {
// Parts to use for labelling
/*
namespace part_map {
    enum {
        PELVIS = 0, L_THIGH, R_THIGH, ABDOMEN, L_SHIN, R_SHIN,
        CHEST,
        L_ANKLE, R_ANKLE, HEAD, L_UPPER_ARM, R_UPPER_ARM,
        L_LOWER_ARM, R_LOWER_ARM, L_HAND, R_HAND,
        _COUNT
    };

    // Mapping from each smpl joint to desired part label number
    // SMPL_JOINT_TO_PART_MAP[smpl joint id] = part label id
    // See Avatar.h, ark::SmplJointType
    const int SMPL_JOINT_TO_PART_MAP[] = {
        PELVIS,  // from ROOT_PELVIS
        L_THIGH, // from L_HIP
        R_THIGH, // from R_HIP
        ABDOMEN, // from SPINE1
        L_SHIN, // from L_KNEE
        R_SHIN, // from R_KNEE
        ABDOMEN, // from SPINE2
        L_ANKLE,
        R_ANKLE,
        CHEST,  // from SPINE3
        L_ANKLE, // from L_FOOT
        R_ANKLE, // from R_FOOT
        HEAD, // from NECK,
        L_UPPER_ARM, // from L_COLLAR
        R_UPPER_ARM, // from R_COLLAR
        HEAD,
        L_UPPER_ARM, // from L_SHOULDER
        R_UPPER_ARM, // from R_SHOULDER
        L_LOWER_ARM, // from L_ELBOW
        R_LOWER_ARM, // from R_ELBOW
        L_HAND, // from L_WRIST
        R_HAND, // from R_WRIST
        L_HAND,
        R_HAND,
    };
}
*/
namespace part_map {
    enum {
        PELVIS = 0,
	THIGH,
	ABDOMEN,
	SHIN,
        CHEST,
	HEAD,
	UPPER_ARM,
        LOWER_ARM,
	HAND_OR_FOOT,
        _COUNT
    };

    // Mapping from each smpl joint to desired part label number
    // SMPL_JOINT_TO_PART_MAP[smpl joint id] = part label id
    // See Avatar.h, ark::SmplJointType
    const int SMPL_JOINT_TO_PART_MAP[] = {
        PELVIS,  // from ROOT_PELVIS
        THIGH, // from L_HIP
        THIGH, // from R_HIP
        ABDOMEN, // from SPINE1
        SHIN, // from L_KNEE
        SHIN, // from R_KNEE
        ABDOMEN, // from SPINE2
        HAND_OR_FOOT,
        HAND_OR_FOOT,
        CHEST,  // from SPINE3
        HAND_OR_FOOT, // from L_FOOT
        HAND_OR_FOOT, // from R_FOOT
        HEAD, // from NECK,
        UPPER_ARM, // from L_COLLAR
        UPPER_ARM, // from R_COLLAR
        HEAD,
        UPPER_ARM, // from L_SHOULDER
        UPPER_ARM, // from R_SHOULDER
        LOWER_ARM, // from L_ELBOW
        LOWER_ARM, // from R_ELBOW
        HAND_OR_FOOT, // from L_WRIST
        HAND_OR_FOOT, // from R_WRIST
        HAND_OR_FOOT,
        HAND_OR_FOOT,
    };
}
}
