#include <opencv2/opencv.hpp>
#include "Edge.h"
#include "EdgeGroup.h"
#include "EBPT.h"

namespace EBPTns {

    class TextureAnalysis {
    public:
        TextureAnalysis();

        EBPT analyzeTexture(const cv::Mat& input_image);

        void setCannyThresholds(double low = 50, double high = 150);
        void setMinEdgeLength(int min_length = 10);
        void setGroupingDistance(int distance = 30);
        std::vector<Edge> extractEdges(const cv::Mat& image);
        std::vector<EdgeGroup> groupEdges(const std::vector<Edge>& edges);

        cv::Mat visualizeEdges(const cv::Mat& image,
            const std::vector<Edge>& edges);
        cv::Mat visualizeGroups(const cv::Mat& image,
            const std::vector<EdgeGroup>& groups);

    private:
        double canny_low_threshold_ = 50.0;
        double canny_high_threshold_ = 150.0;

        int min_edge_length_ = 10;  

        int grouping_distance_ = 30;

        std::vector<std::vector<cv::Point>> findContours(const cv::Mat& edges_image);
        std::vector<cv::Point> simplifyContour(const std::vector<cv::Point>& contour);
        bool shouldGroup(const Edge& edge1, const Edge& edge2) const;
    };
}