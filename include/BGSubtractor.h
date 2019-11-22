#include <opencv2/core.hpp>
namespace ark {
    /** Basic background subtractor */
    class BGSubtractor {
    public:
        /** Create background subtractor with given background image */
        BGSubtractor(cv::Mat background) : background(background) {}

        /** Run background subtraction on image and returns a UINT_8 mask with each number indicating a component
         *  254/255 are background. Optionally, provide comps_by_size to get a list of (pixels in component, compo id in mask) */
        cv::Mat run(const cv::Mat& image, std::vector<std::array<int, 2> >* comps_by_size = nullptr);

        /** Minimum distance to neighbor in background image to consider a point foreground */
        float nnDistThreshRel = 0.005;

        /** Max squared distance to a neighbor, for flood fill */
        float neighbThreshRel = 0.005;

        /** Max allowed number of threads for background subtractor */
        int numThreads = 1;

        /** The background image */
        cv::Mat background;

        /** Current top left and bottom right points of foreground */
        cv::Point topLeft, botRight;
    };
}
