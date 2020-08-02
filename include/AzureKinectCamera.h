#pragma once
#include "Version.h"

#include <opencv2/core.hpp>

#include "DepthCamera.h"

namespace ark {
/**
 * Class defining the behavior of an Azure Kinect (K4A) Camera.
 * Example on how to read from sensor and visualize its output
 * @include SensorIO.cpp
 */
class AzureKinectCamera : public DepthCamera {
   public:
    /**
     * Public constructor initializing the Azure Kinect Camera.
     * @param device_id camera device id. 0 is default.
     * @param wide_fov_mode if true, starts Azure Kinect in wide FOV depth mode
     * @param use_1080p if true, records in 1080p rather than 720p
     * @param scale amount to scale down final image by
     */
    explicit AzureKinectCamera(uint32_t device_id = 0,
                               bool wide_fov_mode = false,
                               bool use_1080p = false, float scale = 1.f);

    /**
     * Destructor for the Azure Kinect Camera.
     */
    ~AzureKinectCamera() override;

    /**
     * Get the camera's model name.
     */
    const std::string getModelName() const override;

    /**
     * Returns the width of the camera frame
     */
    int getWidth() const override;

    /**
     * Returns the height of the camera frame
     */
    int getHeight() const override;

    /**
     * Returns true if an RGB image is available from this camera.
     * @return true if an RGB image is available from this camera.
     */
    bool hasRGBMap() const override;

    /** Preferred frame height */
    const int PREFERRED_FRAME_H = 480;

    /** Shared pointer to Azure Kinect camera instance */
    typedef std::shared_ptr<AzureKinectCamera> Ptr;

   protected:
    /**
     * Gets the new frame from the sensor (implements functionality).
     * Updates xyzMap and ir_map.
     */
    void update(cv::Mat& xyz_map, cv::Mat& rgb_map, cv::Mat& ir_map,
                cv::Mat& amp_map, cv::Mat& flag_map) override;

    // internal storage
    float scale;
    int width, height, scaled_width, scaled_height;

    // Kinect Azure device (k4a_device_t)
    void* k4a_device = NULL;

    // Kinect Azure depth/color transformation (k4a_transformation_t)
    void* k4a_transformation = NULL;

    // Cached XY position multiplier: multiply by current depth to get XY
    // position (k4a_image_t *)
    void* xy_table_cache = NULL;

    const int32_t TIMEOUT_IN_MS = 1000;
};
}  // namespace ark
