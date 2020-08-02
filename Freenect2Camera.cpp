#include "Version.h"
#include "Freenect2Camera.h"

#include <iostream>
#include <numeric>
#include <cmath>
#include <thread>
#include <memory>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

/** Freenect2 Depth Camera Backend **/
namespace ark {
Freenect2Camera::Freenect2Camera(const std::string &serial, bool use_kde,
                                 float scale, bool verbose)
    : _use_kde(use_kde), _serial(serial), _scale(scale), _verbose(verbose) {
    if (!verbose) {
        libfreenect2::setGlobalLogger(nullptr);
    }
    _freenect2 = std::make_unique<libfreenect2::Freenect2>();
    if (_freenect2->enumerateDevices() == 0) {
        std::cerr << "Fatal: No Freenect2 devices found" << std::endl;
        deviceOpenFlag = false;
        return;
    }

#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
    if (_use_kde)
        _pipeline = new libfreenect2::CudaPacketPipeline(-1);
    else
        _pipeline = new libfreenect2::CudaKdePacketPipeline(-1);
#elif defined(LIBFREENECT2_WITH_OPENCL_SUPPORT)
    if (_use_kde)
        _pipeline = new libfreenect2::OpenCLKdePacketPipeline(-1);
    else
        _pipeline = new libfreenect2::OpenCLPacketPipeline(-1);
#elif defined(LIBFREENECT2_WITH_OPENGL_SUPPORT)
    _pipeline = new libfreenect2::OpenGLPacketPipeline();
#else
    _pipeline = new libfreenect2::CPUPacketPipeline();
#endif

    if (_serial.empty()) _serial = _freenect2->getDefaultDeviceSerialNumber();

    if (!_pipeline) return;
    _dev = std::unique_ptr<libfreenect2::Freenect2Device>(
        _freenect2->openDevice(_serial, _pipeline));

    _listener = std::make_unique<libfreenect2::SyncMultiFrameListener>(
        libfreenect2::Frame::Color | libfreenect2::Frame::Depth);

    const int DEFAULT_W = 1920, DEFAULT_H = 1080;
    _scaled_width = _scale * DEFAULT_W;
    _scaled_height = _scale * DEFAULT_H;

    _dev->setColorFrameListener(_listener.get());
    _dev->setIrAndDepthFrameListener(_listener.get());
    if (verbose) {
        std::cout << "Freenect device serial: " << _dev->getSerialNumber()
                  << std::endl;
        std::cout << "Freenect device firmware: " << _dev->getFirmwareVersion()
                  << std::endl;
    }

    if (!_dev->start()) {
        _pipeline = nullptr;
    }

    libfreenect2::Freenect2Device::ColorCameraParams rgb_cam_params =
        _dev->getColorCameraParams();
    _registration = std::make_unique<libfreenect2::Registration>(
        _dev->getIrCameraParams(), rgb_cam_params);

    _xy_table_cache.resize(3, DEFAULT_H * DEFAULT_W);
    for (size_t i = 0; i < DEFAULT_H; ++i) {
        for (size_t j = 0; j < DEFAULT_W; ++j) {
            auto xyz = _xy_table_cache.col(i * DEFAULT_W + j);
            xyz.x() = (j - rgb_cam_params.cx) / rgb_cam_params.fx * 1e-3f;
            xyz.y() = (i - rgb_cam_params.cy) / rgb_cam_params.fy * 1e-3f;
            xyz.z() = 1e-3f;
        }
    }
    fx = rgb_cam_params.fx * _scale;
    cx = rgb_cam_params.cx * _scale;
    fy = rgb_cam_params.fy * _scale;
    cy = rgb_cam_params.cy * _scale;
}

Freenect2Camera::~Freenect2Camera() {
    if (_dev) {
        _dev->stop();
        _dev->close();
    }
}

const std::string Freenect2Camera::getModelName() const {
    return "Kinect V2 (Freenect2)";
}

int Freenect2Camera::getWidth() const { return _scaled_width; }
int Freenect2Camera::getHeight() const { return _scaled_height; }
bool Freenect2Camera::hasRGBMap() const { return true; }

void Freenect2Camera::update(cv::Mat &xyz_map, cv::Mat &rgb_map,
                             cv::Mat &ir_map, cv::Mat &amp_map,
                             cv::Mat &flag_map) {
    if (!_listener->waitForNewFrame(_frames, 10 * 1000)) {
        std::cout << "Freenect2 timeout!" << std::endl;
        badInputFlag = true;
        return;
    }
    badInputFlag = false;
    libfreenect2::Frame *rgb = _frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *depth = _frames[libfreenect2::Frame::Depth];
    libfreenect2::Frame undistorted(depth->width, depth->height, 4);
    libfreenect2::Frame registered(depth->width, depth->height, 4);
    libfreenect2::Frame big_depth(rgb->width, rgb->height + 2, 4);
    std::vector<int> color_to_depth(depth->width * depth->height);
    _registration->apply(rgb, depth, &undistorted, &registered, true,
                         &big_depth, color_to_depth.data());

    const bool NEED_RESIZE = _scale != 1.0f;
    const size_t INPUT_SIZE = rgb->height * rgb->width;

    cv::Mat rgb_tmp, xyz_tmp;
    if (NEED_RESIZE) {
        rgb_tmp.create(rgb->height, rgb->width, CV_8UC3);
        xyz_tmp.create(rgb->height, rgb->width, CV_32FC3);
    }
    Eigen::Map<Eigen::Array<unsigned char, 3, Eigen::Dynamic>>(
        NEED_RESIZE ? rgb_tmp.data : rgb_map.data, 3, INPUT_SIZE) =
        Eigen::Map<Eigen::Array<unsigned char, 4, Eigen::Dynamic>>(rgb->data, 4,
                                                                   INPUT_SIZE)
            .topRows<3>();

    // Depth
    Eigen::Map<Eigen::Array<float, 3, Eigen::Dynamic>> xyz_out(
        reinterpret_cast<float *>(NEED_RESIZE ? xyz_tmp.data : xyz_map.data), 3,
        INPUT_SIZE);
    Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> big_depth_in(
        reinterpret_cast<float *>(big_depth.data) + rgb->width, 1, INPUT_SIZE);
    xyz_out = _xy_table_cache;
    xyz_out.rowwise() *= big_depth_in;
    for (int i = 0; i < INPUT_SIZE; ++i) {
        if (std::isinf(xyz_out(2, i))) {
            xyz_out.col(i).setZero();
        }
    }
    if (NEED_RESIZE) {
        cv::resize(rgb_tmp, rgb_map, cv::Size(_scaled_width, _scaled_height));
        cv::resize(xyz_tmp, xyz_map, cv::Size(_scaled_width, _scaled_height));
    }

    timestamp =
        static_cast<uint64_t>(std::max(rgb->timestamp, depth->timestamp)) *
        125000;

    _listener->release(_frames);
}  // namespace ark
}  // namespace ark
