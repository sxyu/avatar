#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <opencv2/core.hpp>

namespace ark {
    /** ASF/AMC (Acclaim) skeletal model exported from Vicon mocap system.
      * This is the format used by the CMU Mocap dataset (http://mocap.cs.cmu.edu).
      * WARNING: current ASF/AMC file parser limitations and assumptions
      *          - the skeleton must have at most 30 bones + root
      *          - all angles must be in degrees (angle setting is currently not considered)
      *          - all rotations must be specified in XYZ order
      *          - id field assumed to come first in each bonedata section in the ASF file
      *          - direction field assumed to come before length field in each bonedata section
      *          - AMC assumed to be in ":FULLY-SPECIFIED" mode
      * all these constraints are satisfied by data files from the CMU dataset.
      * If there is a need to support other types of files, feel free to change the code.
      */
    class ViconSkeleton {
    public:
        /** Represents a joint */
        struct Joint {
        public:
            Joint(int index, double * data_ptr) :
                index(index), pos(data_ptr), parent(nullptr) { }

            /** Returns the global position of the joint (as Vec3f; may lose precision) */
            cv::Vec3f posVec3f() const;

            /** Returns the global position of the joint (as Vec3d) */
            cv::Vec3d posVec3d() const;

            /** Get the local position (vector from parent joint) of this joint
              * @return local position of non-root joint; global position for root */
            Eigen::Vector3d localPos() const;

            /** Set the local position (vector from parent joint) of this joint
              * sets global position if root */
            void localPos(const Eigen::Vector3d & v);

            /** Get the length of the bone */
            double len() const;

            /** Translate the joint and each joint in its subtree by 'v' */
            void translate(const Eigen::Vector3d & v);

            /** Rotate the bone ending at this joint and each bone in its subtree by 'q'.
              * Does nothing if current joint is root.
              */
            void rotate(const Eigen::Quaterniond & q);

            /** Rotate the bone ending at this joint and each bone in its subtree so that
              * the current bone points in the direction of 'v'.
              * Does nothing if current joint is root.
              */
            void rotate(const Eigen::Vector3d & v);

            /** Scale ONLY the bone ending at this joint by 'scale', translating (but NOT scaling) its children accordingly.
              * Does nothing if this is the root joint. */
            void scaleOne(const double scale);

            /** Scale the bone ending at this joint and each bone in the subtree by 'scale'.
              * Does nothing if this is the root joint. */
            void scale(const double scale);

            /** Rotate and scale the bone ending at this joint and each bone
              * in its subtree so that the current bone has the scale and direction of 'v'.
              * equivalent to (but likely more efficient than) rotate(v) + scale(v.norm()).
              * Does nothing if current joint is root.
              */
            void rotateAndScale(const Eigen::Vector3d & v);

            /** Map to global position of this joint */
            Eigen::Map<Eigen::Vector3d> pos;

            /** Pointer to parent joint */
            Joint * parent;

            /** Pointers to child joints */
            std::vector<Joint *> child;

            /** Name of the joint */
            std::string name;

            /** Index of the joint in the skeleton */
            int index;
        };

        enum class JointType : int {
            // Names of the human bones ending at joints (default naming system in Vicon .asf)
            ROOT = 0, LHIPJOINT, LFEMUR, LTIBIA, LFOOT, LTOES, RHIPJOINT, RFEMUR, RTIBIA,
            RFOOT, RTOES, LOWERBACK, UPPERBACK, THORAX, LOWERNECK, UPPERNECK, HEAD, LCLAVICLE,
            LHUMERUS, LRADIUS, LWRIST, LHAND, LFINGERS, LTHUMB, RCLAVICLE, RHUMERUS, RRADIUS, RWRIST, RHAND, RFINGERS, RTHUMB,

            // alternately, use joint names
            HIP_C = 0, HIP_L, KNEE_L, ANKLE_L, FOOT_L, TOES_L, HIP_R, KNEE_R, ANKLE_R, FOOT_R,
            TOES_R, SPINE1, SPINE2, SHOULDER_C, NECK1, NECK2, SHOULDER_L = 17, ELBOW_L, WRIST1_L, WRIST2_L, HAND_L, FINGERS_L, THUMB_L,
            SHOULDER_R, ELBOW_R, WRIST1_R, WRIST2_R, HAND_R, FINGERS_R, THUMB_R,

            // used for automatically determining the number of joint types
            _COUNT
        };

        /** Construct empty Vicon skeletal model */
        ViconSkeleton();

        /** Load a Vicon skeletal model from ASF/AMC files
          * @param asf ASF file path
          * @param amc AMC file path. If empty, uses skeleton from the ASF file without loading the AMC.
          * @param frame frame number in AMC file to use (1-based indexing; 0 = rest pose)
          * @param length_scale amount to multiply lengths of bones by
          *        (default value converts to meters; will be scaled by 1/units->length in ASF) */
        ViconSkeleton(const std::string & asf, const std::string & amc = "", int frame = 0,
            float length_scale = 0.0254f);

        /** Load Vicon mocap skeleton information from an ASF file. Any loaded ASF/AMC data will be cleared.
          * @param asf ASF file path
          * @param load_rest_pose if true, loads the rest pose to joints
          * @param length_scale amount to multiply lengths of bones by
          *        (default value converts to meters; will be scaled by 1/units->length in ASF)
          */
        void loadASF(const std::string & asf, bool load_rest_pose = true, float length_scale = 0.0254f);

        /** Load Vicon mocap pose info from an AMC file. loadASF must be called before this.
          * Any loaded AMC data will be cleared.
          * @param frame frame number in AMC file to load (1-based indexing; 0 = rest pose) */
        void loadAMC(const std::string & amc, int frame = 0);

        /** Go to the specified frame number in the current AMC file */
        void loadFrame(int frame);

        /** Reset to the resting pose (equivalent to loadFrame(0)) */
        void rest();

        /** Unload all ASF/AMC files, if loaded */
        void unload();
        
        /** Go to next frame defined in the current AMC file 
          * @param num number of frames to advance
          * @param loop if true, loops frames
          * @return true if successfully loaded next frame; false otherwise 
          *        (i.e. the AMC file is loaded and this is not the first frame);
          *        false otherwise */
        bool nextFrame(int num = 1, bool loop = false);

        /** Go to previous frame defined in the current AMC file 
          * @param num number of frames to rewind
          * @param loop if true, loops frames
          * @return true if successfully loaded previous frame
          *        (i.e. the AMC file is loaded and this is not the first frame);
          *        false otherwise */
        bool prevFrame(int num = 1, bool loop = false);

        /** Returns true if an ASF file is loaded */
        bool isASFLoaded() const;

        /** Returns true if an AMC file is loaded */
        bool isAMCLoaded() const;

        /** Get the current frame number in the AMC file (0 if rest pose or no file loaded) */
        int frame() const;

        /** Get the number of frames in the AMC file (0 if not loaded) */
        int numFrames() const;

        /** Get a joint by the name of the bone that ends at it (uses names from ASF file; must be lower case) */
        Joint & getJointByName(const std::string & name);

        /** Approximate SMPL joint positions from Vicon joint positions (heuristic) */
        Eigen::Matrix<double, 3, Eigen::Dynamic> getSmplJoints() const;

        /** Undeformed local position of each joint (ADVANCED) */
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Quaterniond> > refPos; 

    private:
        /** Helper for initializing a frame. Takes in translation and rotation
          * info of each joint and outputs final joint positions to 'trans'
          * @param[in, out] trans translation info + output vector
          * @param[in, out] rot rotation info. may be modified within the function.
          *                 If empty, does not consider rotation.
          */
        void initAMCFrame(std::vector<double> & trans, std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > & rot);

        // ASF data storage
        std::vector<std::vector<std::string>> dof; // degrees of freedom of each joint
        std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > axes, axesInv; // local coordinate system of each joint
        std::unordered_map<std::string, int> jointByName; // maps joint names to joints
        float lengthScale;

        // AMC data storage
        std::vector<std::vector<double>> amcData;

        // general state information
        int curFrame = 0;
        bool asfLoaded = false, amcLoaded = false;

        /** Stores joints */
        std::vector<Joint> joints;

        /** Raw joint position data vector (ADVANCED) */
        std::vector<double> data;
    };
}
