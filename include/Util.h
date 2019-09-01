#pragma once
#include <string>
#include <random>

namespace ark {
    namespace util {
        std::string resolveRootPath(const std::string & root_path);
    }

    // Randomization utilities
    namespace random_util {
        template<class T>
        /** xorshift-based PRNG */
        inline T randint(T lo, T hi) {
            if (hi <= lo) return lo;
            static unsigned long x = std::random_device{}(), y = std::random_device{}(), z = std::random_device{}();
            unsigned long t;
            x ^= x << 16;
            x ^= x >> 5;
            x ^= x << 1;
            t = x;
            x = y;
            y = z;
            z = t ^ x ^ y;
            return z % (hi - lo + 1) + lo;
        }

        /** Uniform distribution */
        float uniform(float min_inc = 0., float max_exc = 1.);

        /** Gaussian distribution */
        float randn(float mean = 0, float variance = 1);
    } // random_util
}
