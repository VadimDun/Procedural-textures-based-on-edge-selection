#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace EBPTns {
    class Edge {
    public:
        Edge() = default;
        Edge(const std::vector<cv::Point>& points);

        void calculateFeatures();
        void applyTransform(const cv::Mat& transform);

        const std::vector<cv::Point>& getPoints() const { return points_; }
        cv::Point getStartPoint() const { return points_.empty() ? cv::Point(0, 0) : points_.front(); }

        float getLength() const { return length_; }
        cv::Point2f getCenter() const { return center_; }

        void setPoints(const std::vector<cv::Point>& points);

    private:
        std::vector<cv::Point> points_;       // Точки ребра

        float length_ = 0.0f;
        cv::Point2f center_;

        void calculateLength();
        void calculateCenter();
    };

}