#pragma once 

#include <vector>
#include <memory>
#include <thread>
#include "opencv2/core.hpp"
#include <Eigen/Core>

#include "Avatar.h"
#include "Calibration.h"

namespace ark {
    class RTree {
    public:
        typedef Eigen::Vector2f Vec2;
        typedef Eigen::Vector2i Vec2i;
        typedef Eigen::VectorXf Distribution;

        struct RNode {
            RNode();
            RNode(const Vec2& u, const Vec2& v, float thresh);

            // Feature data
            Vec2 u, v;
            float thresh;

            // Children indices
            int lnode, rnode;

            // Leaf data
            int leafid;
        };

        /** Create empty RTree with number of different parts */
        explicit RTree(int num_parts);

        /** Load data from path */
        explicit RTree(const std::string & path);

        /** Serialization */
        bool loadFile(const std::string & path);
        bool exportFile(const std::string & path);

        /** Predict distribution for a sample.
         *  Do not call unless model has been trained or loaded */
        Distribution predict(const cv::Mat& depth, const Vec2i& pix);

        /** Predict distribution for all of image. Returns vector of CV_32F Mat 
         *  Do not call unless model has been trained or loaded */
        std::vector<cv::Mat> predict(const cv::Mat& depth);

        /** Train from images and part-masks in OpenARK DataSet format,
         *  with num_images random images and num_points_per_image random pixels
         *  from each image. Do not call train again while training is on-going
         *  on the same RTree. */
        void train(const std::string& depth_dir,
                   const std::string& part_mask_dir,
                   int num_threads = std::thread::hardware_concurrency(),
                   bool verbose = false,
                   int num_images = 30000,
                   int num_points_per_image = 2000,
                   int num_features = 2000,
                   int max_probe_offset = 225, 
                   int min_samples = 100,      // term crit
                   int max_tree_depth = 20,    // term crit 
                   int max_images_loaded = 2000
                   );

        /** Train directly from avatar by rendering simulated images,
         * with num_images random images and
         *  num_points_per_image random pixels from each image.
         *  Do not call train again while training is on-going
         *  on the same RTree. */
        void trainFromAvatar(AvatarModel& avatar_model,
                   AvatarPoseSequence& pose_seq,
                   CameraIntrin& intrin,
                   cv::Size& image_size,
                   int num_threads = std::thread::hardware_concurrency(),
                   bool verbose = false,
                   int num_images = 30000,
                   int num_points_per_image = 2000,
                   int num_features = 2000,
                   int max_probe_offset = 225, 
                   int min_samples = 100,      // term crit
                   int max_tree_depth = 20,     // term crit 
                   const int* part_map = nullptr, // part map
                   int max_images_loaded = 2000
                   );

        std::vector<RNode> nodes;
        std::vector<Distribution> leafData;
        int numParts;

    private:
        Distribution predictRecursive(int nodeid, const cv::Mat& depth, const Vec2i& pix);
    };
}
