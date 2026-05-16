#pragma once
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/ximgproc.hpp>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "Edge.h"
#include "EdgeGroup.h"
#include "SourceGroupInfo.h"

namespace EBPTns {

    struct AnalysisResult {
        std::vector<SourceGroupInfo> source_groups;

        AnalysisResult() = default;

        AnalysisResult(const std::vector<SourceGroupInfo>& sgi)
            : source_groups(sgi){
        }

        bool isValid() const { return !source_groups.empty(); }
    };

    class TextureAnalysis {
    public:
        TextureAnalysis();

        AnalysisResult analyzeTexture(
            const cv::Mat& input_image,
            const std::string& model_path);

        void setMinEdgeLength(int min_length = 10);
        void setSuperpixelParams(int region_size = 30, float ruler = 10.0f, double sp_Thresholds = 0.25);

        bool initializeStructuredDetector(const std::string& model_path);

        int getMinEdgeLength() const { return min_edge_length_; }
        int getSuperpixelRegionSize() const { return superpixel_region_size_; }
        float getSuperpixelRuler() const { return superpixel_ruler_; }
        double getSuperpixelThreshold() const { return superpixel_threshold; }

    private:
        int min_edge_length_ = 10;

        int superpixel_region_size_ = 30;
        float superpixel_ruler_ = 10.0f;
        double superpixel_threshold = 0.25;

        // Пороги для классификации масштабов (в пикселях или радиусе)
        float large_scale_threshold_ = 100.0f;
        float medium_scale_threshold_ = 50.0f;
        float small_scale_threshold_ = 20.0f;

        // Structured Forests
        cv::Ptr<cv::ximgproc::StructuredEdgeDetection> structured_edge_detector_;
        bool is_structured_initialized_ = false;

        std::vector<Edge> extractEdges(const cv::Mat& image, cv::Mat edge_probability_map);

        cv::Mat computeSuperpixels(const cv::Mat& image) const;

        std::vector<std::vector<cv::Point>> findContours(const cv::Mat& edges_image);
        std::vector<cv::Point> simplifyContour(const std::vector<cv::Point>& contour);

        void classifySourceGroups(std::vector<SourceGroupInfo>& source_groups);
        void checkAndAdjustThresholds(std::vector<SourceGroupInfo>& source_groups);
        std::vector<cv::Point> computeHull(const EdgeGroup& group);
    };
}