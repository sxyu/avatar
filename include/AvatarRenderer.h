/** 2D avatar rendering utilities */
#pragma once

#include <vector>
#include <opencv2/core.hpp>

#include "Avatar.h"
#include "Calibration.h"

#include <opencv2/core.hpp>

namespace ark {
    /** 2D depth/parts mask renderer for OpenARK Avatar.
     *  Note: Becomes INVALID if avatar's parameters change,
     *  call renderer.update() or construct a new renderer to
     *  make it valid again
     * */
    class AvatarRenderer {
    public:
        typedef std::pair<float, cv::Vec3i> FaceType;

        /** Construct a 2D renderer for given avatar and depth camera intrinsics */
        AvatarRenderer(const Avatar& ava, const CameraIntrin& intrin);

        /** Get all projected avatar skin points given dpeth camera calibration intrinsics */
        const std::vector<cv::Point2f>& getProjectedPoints() const;

        /** Get all projected avatar joints given dpeth camera calibration intrinsics */
        const std::vector<cv::Point2f>& getProjectedJoints() const;

        /** A list of avatar faces (pairs: (depth, point indices on triangle)) sorted by center depth */
        const std::vector<FaceType>& getOrderedFaces() const;

        /** Render avatar as depth image given image size
         *  Assumes camera is at 0,0,0 and looking in positive z direction */
        cv::Mat renderDepth(const cv::Size& image_size) const;

        /** Render avatar part mask given image size
         *  Part mask is a CV_8U image where pixels assigned to part 0 has value 0, 1 has value 1, etc.
         *  Background pixels have value 255.
         *  Assumes camera is at 0,0,0 and looking in positive z direction
         *  @param part_map optional array of integers specifying part id to assign for each joint; if not given, uses joint id 
         **/
        cv::Mat renderPartMask(const cv::Size& image_size, const int* part_map = nullptr) const;

        /** Render avatar faces given image size
         *  Faces is a CV_32S image where pixels assigned to each face has colors 0, 1, ...
         *  Faces are indexed in the order from getOrderedFaces
         *  Background pixels have value -1.
         *  Assumes camera is at 0,0,0 and looking in positive z direction
         **/
        cv::Mat renderFaces(const cv::Size& image_size) const;

        /** Clear all caches. You must call this whenever avatar parameters change.
         *  Note: this only changes internal cache state and is thus considered a
         *  'const' function in line with other renderer functions */
        void update() const;

    private:
        const Avatar& ava;
        const CameraIntrin& intrin;

        // Cache
        mutable std::vector<cv::Point2f> projectedPoints, projectedJoints;
        mutable std::vector<FaceType> orderedFaces;

    };
}
