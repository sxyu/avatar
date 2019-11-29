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
        // NOTE: ASSUMING width/height <= 32767
        typedef Eigen::Matrix<int16_t, 2, 1> Vec2i;
        /** Probability distribution over numParts parts */
        typedef Eigen::VectorXf Distribution;

        /** Assumed depth of background (meters)
         *  All points not on body will be set to this depth prior to training/detection
         *  (Value is found in RTree.cpp) */
        static const float BACKGROUND_DEPTH;
        // TODO: Possibly, expose background depth as a command line arg.
        // However, I don't think it is very important.

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
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
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

        /** Predict best match for a sample.
         *  Do not call unless model has been trained or loaded */
        uint8_t predictBest(const cv::Mat& depth, const Vec2i& pix);

        /** Predict distribution for all of image. Returns vector of CV_32F Mat 
         *  Do not call unless model has been trained or loaded */
        std::vector<cv::Mat> predict(const cv::Mat& depth);

        /** Predict best match for each pixel in image. Returns CV_8U Mat 
         *  Do not call unless model has been trained or loaded.
         *  Since this is quite performance critical, there are several
         *  heuristic arguments to save computation:
         *  @param interval only computes for pixels with x = y = 0 mod interval
         *  @param top_left top left of ROI to compute (by default, uses entire image)
         *  @param bot_right bottom left of ROI to compute (by default, uses entire image)
         *  @param fill_in_gaps if interval is > 1, setting this to false leaves
         *  pixels not at the interval black. Otherwise fills with the computed pixel
         *  to the top left.  */
        cv::Mat predictBest(const cv::Mat& depth, int num_threads,
                int interval = 1,
                cv::Point top_left = cv::Point(0,0),
                cv::Point bot_right = cv::Point(-1, -1),
                bool fill_in_gaps = true);

        /** Train from images and part-masks in OpenARK DataSet format,
         *  with num_images random images and num_points_per_image random pixels
         *  from each image.
         *  WARNING: Do not call train ON ANY RTree while training is on-going in the
         *  same process, that is, one RTree can be trained at a time in the same process.
         *  TODO: Maybe fix this limitation */
        void train(const std::string& depth_dir,
                   const std::string& part_mask_dir,
                   int num_threads = std::thread::hardware_concurrency(),
                   bool verbose = false,
                   int num_images = 30000,
                   int num_points_per_image = 2000,
                   int num_features = 5000,
                   int num_features_filtered = 200,
                   int max_probe_offset = 225, 
                   int min_samples = 100,      // term crit
                   int max_tree_depth = 20,    // term crit 
                   int min_samples_per_feature = 20,
                   float frac_samples_per_feature = 0.01f,
                   int threshes_per_feature = 15,
                   int max_images_loaded = 50,
                   int mem_limit_mb = 12000,
                   const std::string& train_partial_save_path = ""
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
                   int num_points_per_image = 5000,
                   int num_features = 2000,
                   int num_features_filtered = 200,
                   int max_probe_offset = 225, 
                   int min_samples = 100,      // term crit
                   int max_tree_depth = 20,     // term crit 
                   int min_samples_per_feature = 20,
                   float frac_samples_per_feature = 0.01f,
                   int threshes_per_feature = 15,
                   const std::vector<int>& part_map = {}, // part map
                   int max_images_loaded = 50,
                   int mem_limit_mb = 12000,
                   const std::string& train_partial_save_path = ""
                   );

        /** Re-evaluate leaf distributions on a pre-trained trees
         *  by sampling from simulated images */
        void trainTransfer(AvatarModel& avatar_model,
                   AvatarPoseSequence& pose_seq,
                   CameraIntrin& intrin,
                   cv::Size& image_size,
                   int num_threads = std::thread::hardware_concurrency(),
                   bool verbose = false,
                   int num_images = 10000
                   );

        /** Utility function for post-processing an output body part labels image
         * obtained from predictBest (or manually computed from predict), in order
         * to get higher confidence results.
         * First (optionally) a majority vote is taken in each interval^2 window to get smoother body part labels. 
         * Then only the largest connected component for each body part label is kept
         * @param image Input/output image
         * @param com_pre Stores the centers of mass of each body part
         * at the previous time step; is written to by the function.
         * Should be 2 x numParts, if not matrix is resized automatically
         * by function. If part is not found, x coordinate is set to -1.
         * @param interval size of window to take majority vote of part labels.
         * If interval is 1, no majority vote is used
         * @param dist_to_pre_weight Weight of squared distance to previous
         * center of mass term when finding best cluster for each body part */
        void postProcess(cv::Mat& image,
                Eigen::Matrix<double, 2, Eigen::Dynamic>& com_pre,
                int interval = 1, int num_threads = 1,
                cv::Point top_left = cv::Point(0,0),
                cv::Point bot_right = cv::Point(-1, -1),
                double dist_to_pre_weight = 0.001) const;

        /** Utility for reading a partmap file from an input stream */
        static bool readPartMap(std::istream& is, std::vector<int>& result, int& num_new_parts, int& partmap_type);

        std::vector<RNode, Eigen::aligned_allocator<RNode> > nodes;
        std::vector<Distribution> leafData;
        std::vector<uint8_t> leafBestMatch;

        int numParts;

        std::vector<int> partMap;
        int partMapType = -1;

    private:
        Distribution predictRecursive(int nodeid, const cv::Mat& depth, const Vec2i& pix);
        uint8_t predictRecursiveBest(int nodeid, const cv::Mat& depth, const Vec2i& pix);

        void updateBestMatchTable();
    };
}
