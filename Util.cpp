#include "Util.h"

#include <fstream>
#include <cstdlib>
#include <boost/filesystem.hpp>

namespace ark {
    namespace util {
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
