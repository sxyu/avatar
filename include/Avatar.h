/** OpenARK Avatar Core functionality
 *  SMPL is used but any model with similar data format will work
 **/
#pragma once
#include "Version.h"
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>

#include "GaussianMixture.h"

namespace ark {
class HumanDetector;
struct HumanAvatarUKFModel;

typedef Eigen::Matrix<double, 3, Eigen::Dynamic> CloudType;
typedef Eigen::Matrix<int, 3, Eigen::Dynamic> MeshType;
typedef Eigen::Matrix<double, 2, Eigen::Dynamic> Cloud2DType;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

/** Names for the various skeletal joints in the SMPL model (does not work for
 * other models). For reference only, the avatar will work for any model and
 * does not use this. */
namespace SmplJoint {
enum {
    // TODO: delegate to a skeleton instead

    // BFS Order (topologically sorted)
    ROOT_PELVIS = 0,
    L_HIP,
    R_HIP,
    SPINE1,
    L_KNEE,
    R_KNEE,
    SPINE2,
    L_ANKLE,
    R_ANKLE,
    SPINE3,
    L_FOOT,
    R_FOOT,
    NECK,
    L_COLLAR,
    R_COLLAR,
    HEAD,
    L_SHOULDER,
    R_SHOULDER,
    L_ELBOW,
    R_ELBOW,
    L_WRIST,
    R_WRIST,
    L_HAND,
    R_HAND,

    _COUNT
};
}

/** Represents a generic avatar model, e.g.. SMPL.
 *  This defines the pose/shape of an avatar and cannot be manipulated or viewed
 */
struct AvatarModel {
    /** Create an avatar from the model information in 'model_dir'
     *  @param model_dir path to directory containing model files,
     *                   by default tries to find the one in OpenARK data
     * directory. Must contain model point cloud (model.pcd), skeleton
     * definition (skeleton.txt), and joint regressor (joint_regressor.txt); may
     * also optionally have pose prior (pose_prior.txt), mesh as triangles
     * (mesh.txt), and shape keys aka. blendshapes (shapekey/name.pcd). Joint 0
     * is expected to be root.
     * @param limit_one_joint_per_point only use one assigned joint for each
     * point. This improved performance at the cost of some accuracy.
     */
    explicit AvatarModel(const std::string& model_dir = "",
                         bool limit_one_joint_per_point = false);

    /** Get number of joints */
    inline int numJoints() const { return parent.rows(); }
    /** Get number of skin points */
    inline int numPoints() const { return weights.cols(); }
    /** Get number of shape keys */
    inline int numShapeKeys() const { return keyClouds.cols(); }
    /** Get number of polygon faces */
    inline int numFaces() const { return mesh.cols(); }
    /** Get whether a mesh is available */
    inline bool hasMesh() const { return mesh.cols() > 0; }
    /** Get whether the pose prior model is available */
    inline bool hasPosePrior() const { return posePrior.nComps >= 0; }

    /** Mesh: contains triplets of point indices representing faces (3, num
     * faces) */
    MeshType mesh;

    /** Parent joint index of each joint */
    Eigen::VectorXi parent;

    /** Assigned (weight, joint index) for each point, sorted by
     * descending weight, used for part mask (num points) */
    std::vector<std::vector<std::pair<double, int>>> assignedJoints;

    /** List of points assigned to each joint with weight */
    std::vector<std::vector<std::pair<double, int>>> assignedPoints;

    /** Gaussian Mixture pose prior */
    GaussianMixture posePrior;

    // Advanced data members, for advanced users only
    /** ADVANCED: Base point cloud with positions of each skin point from data
     * file, as a vector (3 * num points). This is kept as a vector to make it
     * easier to add keyClouds which otherwise needs to be a 3D tensor. */
    Eigen::VectorXd baseCloud;

    /** ADVANCED: Shape key (blendshape) data (3*num points, num keys),
     *  each column is vectorized matrix of points x1 y1 z1 x2 y2 z2 ... */
    MatrixType keyClouds;

    /** ADVANCED: Initial joint positions */
    CloudType initialJointPos;

    /** ADVANCED: Joint regressor for recovering joint positions from surface
     * points (num points, num joints) */
    Eigen::SparseMatrix<double> jointRegressor;

    /** ADVANCED: Whether to use the new 'joint shape regressor'
     *  to regress joints
     *  else uses the original joint regressor from SMPL */
    bool useJointShapeRegressor;

    /** ADVANCED: Affine component of joint shape regressor for
     *  recovering joint positions from shape weights directly
     *  (num joints * 3) */
    Eigen::VectorXd jointShapeRegBase;

    /** ADVANCED: Linear transformation component of shape regressor for
     *  recovering joint positions from shape weights directly
     *  (num joints * 3, num shapekeys) */
    Eigen::MatrixXd jointShapeReg;

    /** ADVANCED: Raw LBS Weights (num joints, num points) */
    Eigen::SparseMatrix<double> weights;

    /** ADVANCED: Start index of each joint's assigned points
     *  as in rows of assignWeights (num joints + 1);
     *  terminated with num assignments total */
    Eigen::VectorXi assignStarts;

    /** The directory the avatar's model was imported from */
    const std::string MODEL_DIR;
};

/** Represents a generic avatar instance. The user should construct an
 * AvatarModel first and pass it to HumanAvatar. */
class Avatar {
   public:
    /** Create an avatar by constructing AvatarModel from the model
     *  @see AvatarModel
     */
    explicit Avatar(const AvatarModel& model);

    /** Update the avatar's joints and skin points based on current shape and
     * pose parameters. Must be called at least once after initializing the
     * avatar. WARNING: this is relatively expensive, so don't call it until you
     * really need to get the joints/points of the avatar (updating takes
     * 0.3-0.6 ms typically)
     */
    void update();

    /** Randomize avatar's pose and shape according to PCA (shape) and GMM model
     * (pose). */
    void randomize(bool randomize_pose = true, bool randomize_shape = true,
                   bool randomize_root_pos_rot = true, uint32_t seed = -1);

    /** Random pose from mocap. Requires mocap data (avatar/avatar-mocap) to be
     * downloaded. */
    void randomMocapPose();

    /** Compute the avatar's SMPL pose parameters (axis-angle) */
    Eigen::VectorXd smplParams() const;

    /** Get GMM pdf (likelihood) for current joint rotation parameters */
    double pdf() const;

    /** Try to fit avatar's pose parameters, so that joints are approximately
     * aligned to the given positions. Automatically sets joints prior to
     * joint_pos. */
    void alignToJoints(const CloudType& joint_pos);

    /** The avatar model */
    const AvatarModel& model;

    /** Current point cloud with pose and shape keys both applied (3, num
     * points) */
    CloudType cloud;

    /** Shape-key (aka. blend shape) weights */
    Eigen::VectorXd w;

    /** Root position */
    Eigen::Vector3d p;

    using Mat3Alloc =
        Eigen::aligned_allocator<Eigen::Matrix3d>;  // Alligned matrix allocator

    /** The rotations, stored as 3x3 rotation matrices */
    std::vector<Eigen::Matrix3d, Mat3Alloc> r;

    /** Current joint positions (3, num joints) */
    CloudType jointPos;

    /** Current joint transforms (12, num joints) each 12 is (3, 4) matrix
     * colmajor */
    Eigen::Matrix<double, 12, Eigen::Dynamic> jointTrans;

   private:
    /** INTERNAL for caching use: baseCloud after applying shape keys (3 * num
     * points) */
    Eigen::VectorXd shapedCloudVec;
};

/** A sequence of avatar poses */
struct AvatarPoseSequence {
    /** Create from a sequence file. This should be a binary file,
     *  and there should exist an additional metadata file (<file>.txt)
     *  By default, uses <proj-root>/data/avatar-mocap/cmu-mocap.dat */
    AvatarPoseSequence(const std::string& pose_sequence_path = "");

    /** Load a frame as a raw vector of doubles */
    Eigen::VectorXd getFrame(size_t frame_id) const;

    /** Set the pose of the given avatar class to fit the given
     *  frame in the sequence. Assumes first three values are position,
     *  rest are rotations as quaternions */
    void poseAvatar(Avatar& ava, size_t frame_id) const;

    /** Preload entire file into memory */
    void preload();

    /** Map subsequence name to start frame number */
    std::map<std::string, size_t> subsequences;

    /** Total number of frames */
    size_t numFrames;

    /** Number of doubles (8 bytes) per frame */
    size_t frameSize;

    /** Path to sequence file */
    std::string sequencePath;

   private:
    /** Preloaded data file */
    Eigen::MatrixXd data;
    /** Whether data is preloaded */
    bool preloaded = false;
};
}  // namespace ark
