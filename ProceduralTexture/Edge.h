#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace EBPTns {
    class Edge {
    public:
        Edge() = default;
        Edge(const std::vector<cv::Point>& points);

        void calculateFeatures();
        void applyTransform(const cv::Mat& transform);

        const std::vector<cv::Point>& getPoints() const { return points_; }
        float getLength() const { return length_; }
        cv::Point2f getCenter() const { return center_; }
        float getAngle() const { return angle_; }

        void setPoints(const std::vector<cv::Point>& points);

    private:
        std::vector<cv::Point> points_;
        float length_ = 0.0f;          
        cv::Point2f center_;           
        float angle_ = 0.0f;           

        void calculateLength();
        void calculateCenter();
        void calculateAngle();
    };

}