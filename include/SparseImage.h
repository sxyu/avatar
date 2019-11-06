#include<opencv2/core.hpp>

namespace ark {
    /** Low-memory storage for floating point images with many blank areas */
    struct SparseImage {
        /** Default constructor */
        SparseImage() : rows(0), cols(0) {}

        /** Construct from cv::Mat CV_32F */
        explicit SparseImage(const cv::Mat& image);

        /** Assign from cv::Mat CV_32F */
        SparseImage& operator=(const cv::Mat& image);

        /** Value at row, col */
        float operator()(int y, int x) const;

        /** Value at row, col
         * (for OpenCV Mat compatibility, only float argument works) */
        template<class T>
        T at(int y, int x) const;

        /** Get pixel offset (first nonzero pixel, zero indexed) for row */
        int offset(int y) const;

        /** Get start index (in data vector) for row */
        int start(int y) const;

        /** Convert to cv::Mat CV_32F */
        cv::Mat toMat() const;

        /** Return true if image is empty */
        bool empty() const;

        /** Return cv::Size of image */
        cv::Size size() const;

        /** Compute approx memory usage in bytes */
        size_t memoryUsage() const;

        /** Start indices (internal) */
        std::vector<int> starts;
        /** Data vector (internal) */
        std::vector<float> data;

        /** Image size */
        int rows, cols;
    };
}
