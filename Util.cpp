#include "Util.h"

#include <fstream>
#include <cstdlib>
#include <boost/filesystem.hpp>

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
            static const std::string TEST_PATH = "config/hand-svm/svm.xml";
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
    }
    namespace random_util {
        float uniform(float min_inc, float max_exc) {
            thread_local static std::mt19937 rg(std::random_device{}());
            std::uniform_real_distribution<float> uniform(min_inc, max_exc); 
            return uniform(rg); 
        }

        float randn(float mean, float variance) {
            thread_local static std::mt19937 rg(std::random_device{}());
            std::normal_distribution<float> normal(mean, variance); 
            return normal(rg); 
        }
    }
}
