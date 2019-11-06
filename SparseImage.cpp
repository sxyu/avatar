#include "SparseImage.h"
namespace ark {
    SparseImage::SparseImage(const cv::Mat& image) {
        *this = image;
    }

    SparseImage& SparseImage::operator=(const cv::Mat& image) {
        rows = image.rows; cols = image.cols;
        int curStart = 0;
        starts.reserve(image.rows * 2 + 1);
        data.reserve(image.rows * image.cols / 25);
        for (int i = 0; i < image.rows; ++i) {
            starts.push_back(curStart);
            size_t initDataSz = data.size();
            auto* rowPtr = image.ptr<float>(i);
            int left = std::numeric_limits<int>::max();
            for (int j = 0; j < image.cols; ++j) {
                if (rowPtr[j] != 0.0f) {
                    left = j;
                    break;
                }
            }
            starts.push_back(left);
            if (left == std::numeric_limits<int>::max()) continue;
            int right;
            for (int j = image.cols-1; j >= 0; --j) {
                if (rowPtr[j] != 0.0f) {
                    right = j;
                    break;
                }
            }
            for (int k = left; k <= right; ++k) {
                data.push_back(rowPtr[k]);
            }
            curStart += data.size() - initDataSz;
        }
        starts.push_back(curStart);
        data.shrink_to_fit();
        return *this;
    }

    float SparseImage::operator()(int y, int x) const {
        int left = starts[(y<<1)|1];
        int delta = starts[y<<1];
        int deltaNext = starts[(y+1)<<1];
        int right = left + (deltaNext - delta);
        if (x < left || x >= right) return 0.f;
        return data[delta + x - left];
    }

    template<> float SparseImage::at<float>(int y, int x) const {
        int left = starts[(y<<1)|1];
        int delta = starts[y<<1];
        int deltaNext = starts[(y+1)<<1];
        int right = left + (deltaNext - delta);
        if (x < left || x >= right) return 0.f;
        return data[delta + x - left];
    }

    int SparseImage::offset(int y) const {
        return starts[(y << 1) | 1];
    }

    int SparseImage::start(int y) const {
        return starts[(y << 1)];
    }

    cv::Mat SparseImage::toMat() const {
        cv::Mat result(rows, cols, CV_32FC1);
        for (int i = 0; i < rows; ++i) {
            int left = starts[(i << 1) | 1];
            if (left == std::numeric_limits<int>::max()) continue;
            int delta = starts[i << 1];
            int deltaNext = starts[(i+1) << 1];
            int right = left + (deltaNext - delta);
            auto* rowPtr = result.ptr<float>(i);
            std::fill(rowPtr, rowPtr + left, 0.0f);
            std::fill(rowPtr + right, rowPtr + cols, 0.0f);
            std::copy(data.begin() + delta, data.begin() + deltaNext, rowPtr + left);
        }
        return result;
    }

    bool SparseImage::empty() const {
        return rows == 0 || cols == 0;
    }

    cv::Size SparseImage::size() const {
        return cv::Size(cols, rows);
    }

    size_t SparseImage::memoryUsage() const {
        return sizeof(SparseImage) + data.capacity() * sizeof(float) +
               starts.capacity() * sizeof(int);
    }
}
