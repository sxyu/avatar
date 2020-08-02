#pragma once
#include "Version.h"

#include <opencv2/core.hpp>

#include "DepthCamera.h"

#include <memory>
#include <Eigen/Core>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>

namespace ark {
/**
 * Class defining the behavior of an Kinect V2 Camera using libfreenect2.
 * Example on how to read from sensor and visualize its output
 * @include SensorIO.cpp
 */
class Freenect2Camera : public DepthCamera {
   public:
    /** Freenect2 KinectV2 constructor
     *  @param serial serial number of device to open. Leave empty to use
     * default.
     *  @param use_kde whether to use Kernel Density Estimation (KDE), Lawin et
     * al. ECCV16 only available if Freenect2 built with CUDA or OpenCL
     *  @param scale amount to scale down final image by
     *  @param verbose enable verbose output
     */
    explicit Freenect2Camera(const std::string& serial = "",
                             bool use_kde = false, float scale = 2.f / 3,
                             bool verbose = false);

    /** Clean up method (capture thread) */
    ~Freenect2Camera() override;

    /**
     * Get the camera's model name.
     */
    const std::string getModelName() const override;

    /**
     * Returns the width of the SR300 camera frame
     */
    int getWidth() const override;

    /**
     * Returns the height of the SR300 camera frame
     */
    int getHeight() const override;

    /**
     * Returns true if an RGB image is available from this camera.
     * @return true if an RGB image is available from this camera.
     */
    bool hasRGBMap() const override;

    /** Preferred frame height */
    const int PREFERRED_FRAME_H = 480;

    /** Shared pointer to Freenect2Kinect camera instance */
    typedef std::shared_ptr<Freenect2Camera> Ptr;

   protected:
    /**
     * Gets the new frame from the sensor (implements functionality).
     * Updates xyzMap and ir_map.
     */
    void update(cv::Mat& xyz_map, cv::Mat& rgb_map, cv::Mat& ir_map,
                cv::Mat& amp_map, cv::Mat& flag_map) override;

    /**
     * Initialize the camera, opening channels and resetting to initial
     * configurations
     */
    void initCamera();

    // internal storage
    // * Device info
    // Whether to use Kernel Density Estimation (KDE), Lawin et al. ECCV16
    bool _use_kde;
    // Device serial number
    std::string _serial;
    // Image scaling
    float _scale;
    // Verbose mode
    bool _verbose;

    // Scaled size
    double _scaled_width, _scaled_height;

    // * Context
    std::unique_ptr<libfreenect2::Freenect2> _freenect2;
    std::unique_ptr<libfreenect2::Freenect2Device> _dev;
    libfreenect2::PacketPipeline* _pipeline;
    std::unique_ptr<libfreenect2::Registration> _registration;
    std::unique_ptr<libfreenect2::SyncMultiFrameListener> _listener;
    libfreenect2::FrameMap _frames;

    Eigen::Array<float, 3, Eigen::Dynamic> _xy_table_cache;

    const int32_t TIMEOUT_IN_MS = 1000;
};
}  // namespace ark
