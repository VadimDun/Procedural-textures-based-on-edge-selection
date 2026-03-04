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
        std::vector<Edge> edges;             // Обнаруженные ребра
        std::vector<EdgeGroup> groups;       // Сгруппированные ребра
        cv::Mat edges_visualization;
        cv::Mat groups_visualization;
        cv::Mat edge_probability_map;        // Для Structured Forests (карта вероятностей)
        cv::Mat superpixel_visualization;  // Визуализация суперпикселей
        cv::Mat superpixel_labels;            // Метки суперпикселей
        std::vector<int> group_to_superpixel;

        AnalysisResult() = default;

        // Конструктор для Canny
        AnalysisResult(const EBPT& m,
            const std::vector<Edge>& e,
            const std::vector<EdgeGroup>& g,
            const cv::Mat& ev,
            const cv::Mat& gv)
            : modelEBPT(m), edges(e), groups(g),
            edges_visualization(ev), groups_visualization(gv) {
        }

        // Конструктор для Structured Forests (с картой вероятностей)
        AnalysisResult(const EBPT& m,
            const std::vector<Edge>& e,
            const std::vector<EdgeGroup>& g,
            const cv::Mat& ev,
            const cv::Mat& gv,
            const cv::Mat& prob)
            : modelEBPT(m), edges(e), groups(g),
            edges_visualization(ev), groups_visualization(gv),
            edge_probability_map(prob) {
        }

        bool isValid() const { return !edges.empty() && !groups.empty(); }
    };

    class TextureAnalysis {
    public:
        TextureAnalysis();

        AnalysisResult analyzeTexture(const cv::Mat& input_image);
        AnalysisResult analyzeTextureStructured(const cv::Mat& input_image, const std::string& model_path = "model.yml");
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
        std::vector<EdgeGroup> groupEdges(const std::vector<Edge>& edges);

        cv::Mat visualizeEdges(const cv::Mat& image,
            const std::vector<Edge>& edges);
        cv::Mat visualizeGroups(const cv::Mat& image,
            const std::vector<EdgeGroup>& groups);

        cv::Mat computeSuperpixels(const cv::Mat& image, int region_size, float ruler);
        std::unordered_map<int, std::vector<Edge>> assignEdgesToSuperpixels(const std::vector<Edge>& edges, const cv::Mat& labels);
        cv::Mat visualizeSuperpixels(const cv::Mat& image, const cv::Mat& labels);

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