#pragma once
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/ximgproc.hpp>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "Edge.h"
#include "EdgeGroup.h"
#include "EBPT.h"

namespace EBPTns {

    struct AnalysisResult {
        EBPT modelEBPT;
        cv::Mat superpixel_labels;            // Метки суперпикселей

        AnalysisResult() = default;

        AnalysisResult(const EBPT& m, const cv::Mat& labels)
            : modelEBPT(m), superpixel_labels(labels){
        }

        bool isValid() const { return !modelEBPT.getEdgeGroups().empty() && !superpixel_labels.empty(); }
    };

    class TextureAnalysis {
    public:
        TextureAnalysis();

        AnalysisResult analyzeTextureWithSuperpixelsStructured(
            const cv::Mat& input_image,
            const std::string& model_path,
            int region_size = 30,
            float ruler = 10.0f);

        void setCannyThresholds(double low = 50, double high = 150);
        void setMinEdgeLength(int min_length = 10);
        void setGroupingDistance(int distance = 30);

        void setSuperpixelParams(int region_size = 30, float ruler = 10.0f);
        cv::Mat getSuperpixelMask(const cv::Mat& labels, int superpixel_id);

        std::vector<Edge> extractEdges(const cv::Mat& image);
        std::vector<Edge> extractEdgesStructured(const cv::Mat& image, cv::Mat edge_probability_map);

        cv::Mat computeSuperpixels(const cv::Mat& image, int region_size, float ruler);
        std::unordered_map<int, std::vector<Edge>> assignEdgesToSuperpixels(const std::vector<Edge>& edges, const cv::Mat& labels);

    private:
        double canny_low_threshold_ = 50.0;
        double canny_high_threshold_ = 150.0;

        int min_edge_length_ = 10;  
        int grouping_distance_ = 30;

        int superpixel_region_size_ = 30;
        float superpixel_ruler_ = 10.0f;

        // Structured Forests
        cv::Ptr<cv::ximgproc::StructuredEdgeDetection> structured_edge_detector_;
        bool is_structured_initialized_ = false;

        bool initializeStructuredDetector(const std::string& model_path);

        std::vector<std::vector<cv::Point>> findContours(const cv::Mat& edges_image);
        std::vector<cv::Point> simplifyContour(const std::vector<cv::Point>& contour);
        bool shouldGroup(const Edge& edge1, const Edge& edge2) const;
    };
}