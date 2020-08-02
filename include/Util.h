#pragma once
#include <string>
#include <random>
#include <opencv2/core.hpp>
#include <Eigen/Core>

#define _ARK_ASSERT(x)                                                     \
    do {                                                                   \
        if (!(x)) {                                                        \
            std::cerr << "ark avatar assertion FAILED: \"" << #x << "\" (" \
                      << (bool)(x) << ")\n  at " << __FILE__ << " line "   \
                      << __LINE__ << "\n";                                 \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)
#define _ARK_ASSERT_EQ(x, y)                                                   \
    do {                                                                       \
        if ((x) != (y)) {                                                      \
            std::cerr << "ark avatar assertion FAILED: " << #x << " == " << #y \
                      << " (" << (x) << " != " << (y) << ")\n  at "            \
                      << __FILE__ << " line " << __LINE__ << "\n";             \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
#define _ARK_ASSERT_NE(x, y)                                                   \
    do {                                                                       \
        if ((x) == (y)) {                                                      \
            std::cerr << "ark avatar assertion FAILED: " << #x << " != " << #y \
                      << " (" << (x) << " == " << (y) << ")\n  at "            \
                      << __FILE__ << " line " << __LINE__ << "\n";             \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
#define _ARK_ASSERT_LE(x, y)                                                   \
    do {                                                                       \
        if ((x) > (y)) {                                                       \
            std::cerr << "ark avatar assertion FAILED: " << #x << " <= " << #y \
                      << " (" << (x) << " > " << (y) << ")\n  at " << __FILE__ \
                      << " line " << __LINE__ << "\n";                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
#define _ARK_ASSERT_LT(x, y)                                                  \
    do {                                                                      \
        if ((x) >= (y)) {                                                     \
            std::cerr << "ark avatar assertion FAILED: " << #x << " < " << #y \
                      << " (" << (x) << " >= " << (y) << ")\n  at "           \
                      << __FILE__ << " line " << __LINE__ << "\n";            \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#include <chrono>
#define _ARK_BEGIN_PROFILE \
    auto start = std::chrono::high_resolution_clock::now()
#define _ARK_PROFILE(x)                                                        \
    do {                                                                       \
        double _delta = std::chrono::duration<double, std::milli>(             \
                            std::chrono::high_resolution_clock::now() - start) \
                            .count();                                          \
        printf("%s: %f ms = %f fps\n", #x, _delta, 1e3f / _delta);             \
        start = std::chrono::high_resolution_clock::now();                     \
    } while (false)
#define _ARK_PROFILE_STEPS(x, stp)                                    \
    do {                                                              \
        printf("%s: %f ms / step\n", #x,                              \
               std::chrono::duration<double, std::milli>(             \
                   std::chrono::high_resolution_clock::now() - start) \
                       .count() /                                     \
                   (stp));                                            \
        start = std::chrono::high_resolution_clock::now();            \
    } while (false)

namespace ark {
struct CameraIntrin;
namespace util {
/**
 * Splits a string into components based on a delimiter
 * @param string_in string to split
 * @param delimiters c_str of delimiters to split at
 * @param ignore_empty if true, ignores empty strings
 * @param trim if true, trims whitespaces from each string after splitting
 * @return vector of string components
 */
std::vector<std::string> split(const std::string& string_in,
                               char const* delimiters = " ",
                               bool ignore_empty = false, bool trim = false);

/**
 * Splits a string into components based on a delimiter
 * @param string_in string to split
 * @param delimiters c_str of delimiters to split at
 * @param ignore_empty if true, ignores empty strings
 * @param trim if true, trims whitespaces from each string after splitting
 * @return vector of string components
 */
std::vector<std::string> split(const char* string_in,
                               char const* delimiters = " ",
                               bool ignore_empty = false, bool trim = false);

/** Trims whitespaces (space, newline, etc.) in-place from the left end of the
 * string */
void ltrim(std::string& s);

/** Trims whitespaces (space, newline, etc.) in-place from the right end of the
 * string */
void rtrim(std::string& s);

/** Trims whitespaces (space, newline, etc.) in-place from both ends of the
 * string */
void trim(std::string& s);

/** Convert a string to upper case in-place */
void upper(std::string& s);

/** Convert a string to lower case in-place */
void lower(std::string& s);

std::string resolveRootPath(const std::string& root_path);

/**
 * Get the color at index 'index' of the built-in palette
 * Used to map integers to colors.
 * @param color_index index of color
 * @param bgr if true, color is returned in BGR order instead of RGB (default
 * true)
 * @return color in Vec3b format
 */
cv::Vec3b paletteColor(int color_index, bool bgr = true);

/** Create table of num_colors colors, shape (3, num_colors) */
Eigen::Matrix<float, 3, Eigen::Dynamic> paletteColorTable(int num_colors,
                                                          bool bgr = true);

template <class T>
/** Write binary to ostream */
inline void write_bin(std::ostream& os, T val) {
    os.write(reinterpret_cast<char*>(&val), sizeof(T));
}

template <class T>
/** Read binary from istream */
inline void read_bin(std::istream& is, T& val) {
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
}

/** Estimate pinhole camera intrinsics from xyz_map (by solving OLS)
 *  @return (fx, cx, fy, cy)
 */
cv::Vec4d getCameraIntrinFromXYZ(const cv::Mat& xyz_map);

/** Read a '.depth' raw depth map file into an OpenCV Mat
 *  @param allow_exr if true, checks if the file format is exr,
 *                   in which case does cv::imread instead
 *  */
void readDepth(const std::string& path, cv::Mat& m, bool allow_exr = true);

/** Read a '.depth' raw depth map file into an OpenCV Mat as XYZ map;
 *  if image already has 3 channels then reads directly
 *  @param allow_exr if true, checks if the file format is exr,
 *                   in which case does cv::imread instead
 *  */
void readXYZ(const std::string& path, cv::Mat& m, const CameraIntrin& intrin,
             bool allow_exr = true);

/** Write a .depth raw depth map file from an OpenCV Mat */
void writeDepth(const std::string& image_path, cv::Mat& depth_map);

// Angle-axis to rotation matrix using custom implementation
template <class T, int Option = Eigen::ColMajor>
inline Eigen::Matrix<T, 3, 3, Option> rodrigues(
    const Eigen::Ref<Eigen::Matrix<T, 3, 1>>& vec) {
    const T theta = vec.norm();
    const Eigen::Matrix<T, 3, 3, Option> eye =
        Eigen::Matrix<T, 3, 3, Option>::Identity();

    if (std::fabs(theta) < 1e-5f)
        return eye;
    else {
        const T c = std::cos(theta);
        const T s = std::sin(theta);
        const Eigen::Matrix<T, 3, 1> r = vec / theta;
        Eigen::Matrix<T, 3, 3, Option> skew;
        skew << 0, -r.z(), r.y(), r.z(), 0, -r.x(), -r.y(), r.x(), 0;
        return c * eye + (1 - c) * r * r.transpose() + s * skew;
    }
}

// Affine transformation matrix (hopefully) faster multiplication
// bottom row omitted
template <class T, int Option = Eigen::ColMajor>
inline void mulAffine(const Eigen::Ref<const Eigen::Matrix<T, 3, 4, Option>>& a,
                      Eigen::Ref<Eigen::Matrix<T, 3, 4, Option>> b) {
    b.template leftCols<3>() =
        a.template leftCols<3>() * b.template leftCols<3>();
    b.template rightCols<1>() =
        a.template rightCols<1>() +
        a.template leftCols<3>() * b.template rightCols<1>();
}

// Affine transformation matrix 'in-place' inverse
template <class T, int Option = Eigen::ColMajor>
inline void invAffine(
    const Eigen::Ref<const Eigen::Matrix<T, 3, 4, Option>>& a) {
    a.template leftCols<3>() = a.template leftCols<3>().inverse();
    a.template rightCols<1>() =
        -a.template leftCols<3>() * a.template rightCols<1>();
}

// Homogeneous transformation matrix in-place inverse
template <class T, int Option = Eigen::ColMajor>
inline void invHomogeneous(
    const Eigen::Ref<const Eigen::Matrix<T, 3, 4, Option>>& a) {
    a.template leftCols<3>().transposeInPlace();
    a.template rightCols<1>() =
        -a.template leftCols<3>() * a.template rightCols<1>();
}
}  // namespace util

// Randomization utilities
namespace random_util {
template <class T>
/** xorshift-based PRNG */
inline T randint(T lo, T hi) {
    if (hi <= lo) return lo;
    static thread_local unsigned long x = std::random_device{}(),
                                      y = std::random_device{}(),
                                      z = std::random_device{}();
    thread_local unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;
    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;
    return z % (hi - lo + 1) + lo;
}

template <class T, class A>
/** Choose k elements from a vector */
std::vector<T, A> choose(std::vector<T, A>& source, size_t k) {
    std::vector<T, A> out;
    for (size_t j = 0; j < std::min(k, source.size()); ++j) {
        int r = randint(j, source.size() - 1);
        out.push_back(source[r]);
        std::swap(source[j], source[r]);
    }
    return out;
}

template <class T, class A>
/** Choose k elements from an interval (inclusive on left, exclusive right) of a
   vector */
std::vector<T, A> choose(std::vector<T, A>& source, size_t l, size_t r,
                         size_t k) {
    std::vector<T, A> out;
    for (size_t j = l; j < std::min(l + k, r); ++j) {
        int ran = randint(j, r - 1);
        out.push_back(source[ran]);
        std::swap(source[j], source[ran]);
    }
    return out;
}

/** Uniform distribution */
float uniform(float min_inc = 0., float max_exc = 1.);

/** Gaussian distribution */
float randn(float mean = 0, float variance = 1);

/** Uniform distribution with provided rng */
float uniform(std::mt19937& rg, float min_inc = 0., float max_exc = 1.);

/** Gaussian distribution with provided rng */
float randn(std::mt19937& rg, float mean = 0, float variance = 1);
}  // namespace random_util
}  // namespace ark
