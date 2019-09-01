#pragma once

#define AZURE_KINECT_ENABLED
#define OPENARK_CAMERA_TYPE "azurekinect"
//#define RSSDK2_ENABLED
//#define OPENARK_CAMERA_TYPE "realsense"
//#define RSSDK_ENABLED
//#define OPENARK_CAMERA_TYPE "sr300"
//#define PMDSDK_ENABLED
//#define OPENARK_CAMERA_TYPE "pmd" 

#define OPENARK_VERSION_MAJOR 0
#define OPENARK_VERSION_MINOR 9
#define OPENARK_VERSION_PATCH 4
#define ARK_STRINGIFY2(X) #X
#define ARK_STRINGIFY(X) ARK_STRINGIFY2(X)

// Uncomment to enable debug code
#define ARK_DEBUG

// Constants
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// Custom assertion statement that prints out a message
#if !defined(NDEBUG) || defined(ARK_DEBUG)
#   define ARK_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
             std::cerr << "OpenARK assertion '" << (#condition) << "' failed in " << \
              __FILE__ << " at line " << __LINE__ << ": " << message << "\n"; \
            std::terminate(); \
        } \
    } while (false)
#else
// disable assert if not debugging
#  define ARK_ASSERT(condition, message) do { } while (false)
#endif

// Necessary for typedefs
#include <opencv2/core/types.hpp>

// OpenARK namespace
namespace ark {
    // OpenARK version number (modify in CMakeLists.txt)
    static const char * VERSION = ARK_STRINGIFY(OPENARK_VERSION_MAJOR)
                                  "."
                                  ARK_STRINGIFY(OPENARK_VERSION_MINOR)
                                  "."
                                  ARK_STRINGIFY(OPENARK_VERSION_PATCH);

    // Typedefs for common types
    typedef cv::Point Point;
    typedef cv::Point2i Point2i;
    typedef cv::Point2f Point2f;
    typedef cv::Point2d Point2d;
    typedef cv::Vec2f Vec2f;
    typedef cv::Vec2d Vec2d;
    typedef cv::Vec2i Vec2i;
    typedef cv::Vec3b Vec3b;
    typedef cv::Vec3f Vec3f;
    typedef cv::Vec3d Vec3d;
    typedef cv::Vec3i Vec3i;
}
