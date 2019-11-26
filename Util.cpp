#include "Util.h"

#include <fstream>
#include <cstdlib>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <opencv2/imgcodecs.hpp>

#include "Calibration.h"

namespace ark {
    namespace util {
        std::vector<std::string> split(const std::string & string_in, char const * delimiters,
            bool ignore_empty, bool trim) {
            char * buffer = new char[string_in.size() + 1];
            strcpy(buffer, string_in.c_str());
            std::vector<std::string> output;
            for (char * token = strtok(buffer, delimiters);
                token != NULL; token = strtok(NULL, delimiters))
            {
                output.emplace_back(token);
                util::trim(*output.rbegin());
                if (ignore_empty && output.rbegin()->empty()) output.pop_back();
            }
            delete[] buffer;
            return output;
        }

        std::vector<std::string> split(const char * string_in, char const * delimiters,
            bool ignore_empty, bool trim) {
            return split(std::string(string_in), delimiters, ignore_empty, trim);
        }

        // trimming functions from: https://stackoverflow.com/questions/216823/
        // trim from start (in place)
        void ltrim(std::string & s) {
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
                return !std::isspace(ch);
            }));
        }

        // trim from end (in place)
        void rtrim(std::string & s) {
            s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
                return !std::isspace(ch);
            }).base(), s.end());
        }

        // trim from both ends (in place)
        void trim(std::string & s) {
            ltrim(s);
            rtrim(s);
        }

        void upper(std::string & s)
        {
            for (size_t i = 0; i < s.size(); ++i)
                s[i] = std::toupper(s[i]);
        }

        void lower(std::string & s)
        {
            for (size_t i = 0; i < s.size(); ++i)
                s[i] = std::tolower(s[i]);
        }

        std::string resolveRootPath(const std::string & root_path)
        {
            static const std::string TEST_PATH = "data/avatar-model/extract.py";
            static const int MAX_LEVELS = 3;
            static std::string rootDir = "\n";
            if (rootDir == "\n") {
                rootDir.clear();
                const char * env = std::getenv("OPENARK_DIR");
                if (env) {
                    // use environmental variable if exists and works
                    rootDir = env;

                    // auto append slash
                    if (!rootDir.empty() && rootDir.back() != '/' && rootDir.back() != '\\')
                        rootDir.push_back('/');

                    std::ifstream test_ifs(rootDir + TEST_PATH);
                    if (!test_ifs) rootDir.clear();
                }

                const char * env2 = std::getenv("SMPLSYNTH_DIR");
                if (env2) {
                    // use environmental variable if exists and works
                    rootDir = env2;

                    // auto append slash
                    if (!rootDir.empty() && rootDir.back() != '/' && rootDir.back() != '\\')
                        rootDir.push_back('/');

                    std::ifstream test_ifs(rootDir + TEST_PATH);
                    if (!test_ifs) rootDir.clear();
                }

                // else check current directory and parents
                if (rootDir.empty()) {
                    for (int i = 0; i < MAX_LEVELS; ++i) {
                        std::ifstream test_ifs(rootDir + TEST_PATH);
                        if (test_ifs) break;
                        rootDir.append("../");
                    }
                }
            }
            typedef boost::filesystem::path path;
            return (path(rootDir) / path(root_path)).string();
        }
        cv::Vec3b paletteColor(int color_index, bool bgr)
        {
            using cv::Vec3b;
            static const Vec3b palette[] = {
                Vec3b(0, 220, 255), Vec3b(177, 13, 201), Vec3b(94, 255, 34),
                Vec3b(54, 65, 255), Vec3b(64, 255, 255), Vec3b(217, 116, 0),
                Vec3b(27, 133, 255), Vec3b(190, 18, 240), Vec3b(20, 31, 210),
                Vec3b(75, 20, 133), Vec3b(255, 219, 127), Vec3b(204, 204, 57),
                Vec3b(112, 153, 61), Vec3b(64, 204, 46), Vec3b(112, 255, 1),
                Vec3b(170, 170, 170), Vec3b(225, 30, 42)
            };

            Vec3b color = palette[color_index % (int)(sizeof palette / sizeof palette[0])];
            return bgr ? color : Vec3b(color[2], color[1], color[0]);
        }

        cv::Vec4d getCameraIntrinFromXYZ(const cv::Mat & xyz_map)
        {
            int rows = xyz_map.rows, cols = xyz_map.cols;
            Eigen::MatrixXd A(rows * cols, 2);
            Eigen::MatrixXd b(rows * cols, 1);
            cv::Vec4d result;

            // fx cx
            const cv::Vec3f * ptr;
            for (int r = 0; r < rows; ++r) {
                ptr = xyz_map.ptr<cv::Vec3f>(r);
                for (int c = 0; c < cols; ++c) {
                    const int i = r * cols + c;
                    A(i, 0) = ptr[c][0];
                    A(i, 1) = ptr[c][2];
                    b(i) = c * ptr[c][2];
                }
            }

            Eigen::Vector2d wx = A.colPivHouseholderQr().solve(b);
            result[0] = wx[0];
            result[1] = wx[1];

            // fy cy
            for (int r = 0; r < rows; ++r) {
                ptr = xyz_map.ptr<cv::Vec3f>(r);
                for (int c = 0; c < cols; ++c) {
                    const int i = r * cols + c;
                    A(i, 0) = ptr[c][1];
                    A(i, 1) = ptr[c][2];
                    b(i) = r * ptr[c][2];
                }
            }

            Eigen::Vector2d wy = A.colPivHouseholderQr().solve(b);
            result[2] = wy[0];
            result[3] = wy[1];
            return result;
        }

        void readDepth(const std::string & path, cv::Mat & m, bool allow_exr) {
            if (allow_exr && path.size() > 4 &&
                    !path.compare(path.size()-4, path.size(), ".exr")) {
                // Read .exr instead
                m = cv::imread(path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
                return;
            }
            std::ifstream ifs(path, std::ios::binary | std::ios::in);

            ushort wid, hi;
            util::read_bin(ifs, hi);
            util::read_bin(ifs, wid);

            m = cv::Mat::zeros(hi, wid, CV_32FC1);

            int zr = 0;
            for (int i = 0; i < hi; ++i) {
                float * ptr = m.ptr<float>(i);
                for (int j = 0; j < wid; ++j) {
                    if (zr) --zr;
                    else {
                        if (!ifs) break;
                        float x; util::read_bin(ifs, x);
                        if (x >= 0) {
                            ptr[j] = x;
                        }
                        else {
                            zr = (int)(-x) - 1;
                        }
                    }
                }
            }
        }

        void readXYZ(const std::string & path, cv::Mat & m,
                const CameraIntrin& intrin, bool allow_exr) {
            readDepth(path, m, allow_exr);
            if (!m.empty() && m.channels() == 1) {
                m = intrin.depthToXYZ(m);
            }
        }

        void writeDepth(const std::string & image_path, cv::Mat & depth_map) {
            std::ofstream ofsd(image_path, std::ios::binary | std::ios::out);

            if (ofsd) {
                util::write_bin(ofsd, (ushort)depth_map.rows);
                util::write_bin(ofsd, (ushort)depth_map.cols);

                int zrun = 0;
                for (int i = 0; i < depth_map.rows; ++i)
                {
                    const float * ptr = depth_map.ptr<float>(i);
                    for (int j = 0; j < depth_map.cols; ++j)
                    {
                        if (ptr[j] == 0) {
                            ++zrun;
                            continue;
                        }
                        else {
                            if (zrun >= 1) {
                                util::write_bin(ofsd, (float)(-zrun));
                            }
                            zrun = 0;
                            util::write_bin(ofsd, ptr[j]);// util::write_bin(ofsd, ptr[j][1]); writeBinary(ofsd, ptr[j][2]);
                        }
                    }
                }

                ofsd.close();
            }
        }
    }

    namespace random_util {
        float uniform(float min_inc, float max_exc) {
            thread_local static std::mt19937 rg(std::random_device{}());
            return uniform(rg, min_inc, max_exc);
        }

        float randn(float mean, float variance) {
            thread_local static std::mt19937 rg(std::random_device{}());
            return randn(rg, mean, variance);
        }

        float uniform(std::mt19937& rg, float min_inc, float max_exc) {
            std::uniform_real_distribution<float> uniform(min_inc, max_exc);
            return uniform(rg);
        }

        float randn(std::mt19937& rg, float mean, float variance) {
            std::normal_distribution<float> normal(mean, variance);
            return normal(rg);
        }
    }
}
